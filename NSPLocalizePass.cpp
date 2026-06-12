//===- NSPLocalizePass.cpp - NSP localization
//------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//
//
// NSP localization pass.
//
// This pass performs the tensor-semantic half of the NSP bring-up pipeline for
// multi-core execution. It consumes declarative shard annotations and rewrites
// supported computations into shard-local tensor computations, but it does not
// materialize those local tiles into destination memrefs.
//
// The current implementation targets the simplest sharded elementwise patterns
// (e.g. vadd expressed as linalg.generic) for ranked tensors up to
// rank 4 on the identity-map path, plus selected rank-2 broadcast patterns,
// and rewrites them into:
//   - per-core local slices using shard.all_slice
//   - local compute (linalg.generic) on the sliced tensors
//   - optional reconstitution using shard.all_gather when collectives are
//     enabled and a global tensor value must be preserved
//
// In non-collective mode, this pass makes the hand-off to the later
// NSPMaterializePass explicit by creating an `nsp.materialize_tile` op.
// That op anchors the final destination materialization without keeping the
// original global linalg.generic alive in the IR.
//
// Expected IR (high level):
//   - A shard.grid symbol exists in the module (e.g. @nsp).
//   - Sharding descriptors (!shard.sharding) are produced and attached/used
//     as part of the propagation/planning flow.
//   - In non-collective mode, computations with an identifiable
//     materialize_in_destination sink are rewritten into:
//         * shard-local tensor compute
//         * nsp.materialize_tile for the final destination hand-off
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

#include <functional>
#include <iterator>
#include <optional>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mlir/IR/BuiltinAttributes.h"

// Shard dialect ops/types.
#include "mlir/Dialect/Shard/IR/ShardDialect.h"
#include "mlir/Dialect/Shard/IR/ShardOps.h"

// NSP dialect ops/types.
#include "hexagon/Dialect/NSP/IR/NSPDialect.h"
#include "hexagon/Dialect/NSP/IR/NSPOps.h"

namespace mlir {
namespace hexagon {

namespace {

//===----------------------------------------------------------------------===//
// Localize: build tensor-based local linalg.generic and attach attributes
//===----------------------------------------------------------------------===//
static FailureOr<linalg::GenericOp>
localizeGenericToTensor(OpBuilder &b, linalg::GenericOp g,
                        RankedTensorType localTy, ArrayRef<Value> localInputs,
                        Value oldInit, int64_t splitAxis) {
  Location loc = g.getLoc();

  // Reuse the existing init tensor when it already matches the local type.
  Value outLocalInit = oldInit;
  auto oldInitTy = dyn_cast<RankedTensorType>(oldInit.getType());
  if (!oldInitTy || oldInitTy != localTy) {
    outLocalInit = tensor::EmptyOp::create(b, loc, localTy.getShape(),
                                           localTy.getElementType());
  }

  auto localizedGeneric = linalg::GenericOp::create(
      b, loc,
      /*resultTensorTypes=*/TypeRange{localTy},
      /*inputs=*/ValueRange{localInputs},
      /*outputs=*/ValueRange{outLocalInit},
      /*indexingMaps=*/g.getIndexingMapsArray(),
      /*iteratorTypes=*/g.getIteratorTypesArray(),
      /*doc=*/StringRef(), /*libraryCall=*/StringRef());

  // Clone the region body instead of moving it. The original global generic
  // must stay intact until NSPMaterializePass reconnects the localized result
  // to the final destination.
  Region &dstRegion = localizedGeneric.getRegion();
  dstRegion.getBlocks().clear();

  Block &srcBlock = g.getRegion().front();
  Block *dstBlock = new Block();
  dstRegion.push_back(dstBlock);

  for (BlockArgument a : srcBlock.getArguments())
    dstBlock->addArgument(a.getType(), a.getLoc());

  IRMapping map;
  for (auto [sa, da] :
       llvm::zip(srcBlock.getArguments(), dstBlock->getArguments()))
    map.map(sa, da);

  OpBuilder nb = OpBuilder::atBlockEnd(dstBlock);
  for (Operation &op : srcBlock.getOperations())
    nb.clone(op, map);

  SmallVector<int64_t> tileShape(localTy.getShape().begin(),
                                 localTy.getShape().end());
  for (int64_t dim : tileShape) {
    if (ShapedType::isDynamic(dim) || dim <= 0)
      return failure();
  }

  localizedGeneric->setAttr("nsp.localized", b.getUnitAttr());
  localizedGeneric->setAttr("nsp.materialize_tile_shape",
                            b.getDenseI64ArrayAttr(tileShape));
  localizedGeneric->setAttr("nsp.materialize_split_axis",
                            b.getI64IntegerAttr(splitAxis));

  return localizedGeneric;
}

//===----------------------------------------------------------------------===//
// Broadcast-aware generic helpers
//===----------------------------------------------------------------------===//

static constexpr int64_t kMaxSupportedIdentityElementwiseRank = 4;

/// Return true iff the generic identity-map elementwise path supports `rank`.
///
/// Rank-2 broadcast/reduction-specific helpers remain intentionally separate;
/// this predicate only controls the shape-preserving identity-map path used by
/// elementwise kernels such as GELU.
static bool isSupportedIdentityElementwiseRank(int64_t rank) {
  return rank >= 1 && rank <= kMaxSupportedIdentityElementwiseRank;
}

/// Return true iff `map` is the rank-2 identity map:
///   (d0, d1) -> (d0, d1)
static bool isIdentity2DMap(AffineMap map) {
  if (!map || map.getNumResults() != 2)
    return false;

  auto d0 = dyn_cast<AffineDimExpr>(map.getResult(0));
  auto d1 = dyn_cast<AffineDimExpr>(map.getResult(1));
  if (!d0 || !d1)
    return false;

  return d0.getPosition() == 0 && d1.getPosition() == 1;
}

/// Return true iff `map` is the row-broadcast projection:
///   (d0, d1) -> (d0)
static bool isRowBroadcastMap(AffineMap map) {
  if (!map || map.getNumResults() != 1)
    return false;

  auto d0 = dyn_cast<AffineDimExpr>(map.getResult(0));
  if (!d0)
    return false;

  return d0.getPosition() == 0;
}

/// Return true iff `map` is the column-broadcast projection:
///   (d0, d1) -> (d1)
static bool isColumnBroadcastMap(AffineMap map) {
  if (!map || map.getNumResults() != 1)
    return false;

  auto d1 = dyn_cast<AffineDimExpr>(map.getResult(0));
  if (!d1)
    return false;

  return d1.getPosition() == 1;
}

/// Return true iff `g` matches a supported rank-2 broadcast elementwise
/// generic:
///   - rank-2 result
///   - all-parallel iterators
///   - output map is identity 2D
///   - each input is either:
///       * rank-2 with identity 2D map,
///       * rank-1 with row-broadcast map (d0, d1) -> (d0), or
///       * rank-1 with column-broadcast map (d0, d1) -> (d1)
static bool isSupportedBroadcast2DGeneric(linalg::GenericOp g,
                                          RankedTensorType outResTy,
                                          ArrayRef<RankedTensorType> inputTys) {
  if (!outResTy || outResTy.getRank() != 2)
    return false;

  if (g.getNumLoops() != 2)
    return false;

  auto iters = g.getIteratorTypesArray();
  if (iters.size() != 2)
    return false;
  if (!llvm::all_of(iters, [](mlir::utils::IteratorType it) {
        return it == mlir::utils::IteratorType::parallel;
      }))
    return false;

  if (g.getNumDpsInits() != 1)
    return false;

  auto maps = g.getIndexingMapsArray();
  if ((int64_t)maps.size() != g.getNumDpsInputs() + g.getNumDpsInits())
    return false;

  // Output must be identity 2D.
  if (!isIdentity2DMap(maps.back()))
    return false;

  // Inputs may be:
  //   - rank-2 + identity 2D
  //   - rank-1 + row-broadcast projection
  //   - rank-1 + column-broadcast projection
  for (int64_t i = 0, e = static_cast<int64_t>(inputTys.size()); i < e; ++i) {
    RankedTensorType inTy = inputTys[i];
    if (!inTy)
      return false;

    AffineMap map = maps[i];
    if (inTy.getRank() == 2) {
      if (!isIdentity2DMap(map))
        return false;
      continue;
    }

    if (inTy.getRank() == 1) {
      if (!isRowBroadcastMap(map) && !isColumnBroadcastMap(map))
        return false;
      continue;
    }

    return false;
  }

  return true;
}

//===----------------------------------------------------------------------===//
// Drop stale shard.shard wrappers after localization has consumed sharding
// annotations.
//===----------------------------------------------------------------------===//
static void stripShardAnnotationWrappers(ModuleOp module) {
  SmallVector<Operation *> eraseList;

  module.walk([&](mlir::shard::ShardOp op) {
    if (op->getNumOperands() < 1 || op->getNumResults() < 1)
      return;

    Value src = op->getOperand(0);
    Value result = op->getResult(0);
    result.replaceAllUsesWith(src);
    eraseList.push_back(op);
  });

  for (Operation *op : eraseList)
    op->erase();

  eraseList.clear();

  module.walk([&](mlir::shard::ShardingOp op) {
    if (op->getNumResults() == 1 && op->getResult(0).use_empty())
      eraseList.push_back(op);
  });

  for (Operation *op : eraseList)
    op->erase();
}

//===----------------------------------------------------------------------===//
// NSP localization pass.
//===----------------------------------------------------------------------===//
struct NSPLocalizePass
    : public PassWrapper<NSPLocalizePass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NSPLocalizePass)

  NSPLocalizePass() : NSPLocalizePass(/*allowCollectives=*/false) {}

  /// Constructor used by createNSPLocalizePass(bool).
  explicit NSPLocalizePass(bool allow)
      : PassWrapper(),
        allowCollectives(*this, "allow-collectives",
                         llvm::cl::desc("Allow NSPLocalizePass to insert shard "
                                        "collectives (e.g. all_gather)"),
                         llvm::cl::init(allow)) {}

  /// PassWrapper's default clonePass implementation relies on a copy
  /// constructor. Pass::Option is not copyable, so we must explicitly define
  /// a copy constructor that re-initializes the options from values.
  NSPLocalizePass(const NSPLocalizePass &other)
      : NSPLocalizePass(static_cast<bool>(other.allowCollectives)) {}

  StringRef getArgument() const final { return "nsp-localize"; }

  StringRef getDescription() const final {
    return "Localize supported sharded computations into tensor-semantic "
           "shard-local compute";
  }

  // When enabled, this pass is allowed to materialize collectives (e.g.
  // shard.all_gather) as part of the bring-up SPMDization.
  Option<bool> allowCollectives;

  void runOnOperation() final {
    MLIRContext *ctx = &getContext();

    ctx->getOrLoadDialect<mlir::shard::ShardDialect>();
    ctx->getOrLoadDialect<mlir::affine::AffineDialect>();
    ctx->getOrLoadDialect<mlir::arith::ArithDialect>();
    ctx->getOrLoadDialect<mlir::bufferization::BufferizationDialect>();
    ctx->getOrLoadDialect<mlir::func::FuncDialect>();
    ctx->getOrLoadDialect<mlir::linalg::LinalgDialect>();
    ctx->getOrLoadDialect<mlir::memref::MemRefDialect>();
    ctx->getOrLoadDialect<mlir::scf::SCFDialect>();
    ctx->getOrLoadDialect<mlir::tensor::TensorDialect>();
    ctx->getOrLoadDialect<mlir::hexagon::nsp::NSPDialect>();

    ModuleOp module = getOperation();

    // Validate that the expected grid symbol exists.
    // The planner currently creates shard.grid @nsp.
    auto grid = module.lookupSymbol<mlir::shard::GridOp>("nsp");
    if (!grid) {
      module.emitError()
          << "NSPLocalizePass expected a 'shard.grid' symbol named '@nsp' "
             "in the module, but none was found. "
             "Ensure NSP shard planning ran and created the grid.";
      signalPassFailure();
      return;
    }

    // Read grid shape (assume 1D grid for now).
    auto shape = grid.getShape();
    if (shape.empty()) {
      module.emitError() << "shard.grid '@nsp' has an empty shape";
      signalPassFailure();
      return;
    }

    const int64_t numShards = shape.front();
    if (numShards <= 0) {
      module.emitError() << "invalid shard grid size (shape[0]) = "
                         << numShards;
      signalPassFailure();
      return;
    }

    auto isNspDistributedLoop = [](mlir::scf::ForOp forOp) -> bool {
      return forOp && (forOp->hasAttr("nsp.distributed") ||
                       forOp->hasAttr("nsp.distribution"));
    };

    auto hasNspDistributedLoopAncestor = [&](Operation *op) -> bool {
      constexpr unsigned maxDepth = 100; // Reasonable nesting limit
      unsigned depth = 0;
      for (Operation *parent = op ? op->getParentOp() : nullptr; parent;
           parent = parent->getParentOp()) {
        if (++depth > maxDepth) {
          op->emitWarning() << "NSPLocalize: excessive nesting depth while "
                               "checking for distributed loop ancestor";
          return false;
        }
        if (auto parentFor = dyn_cast<mlir::scf::ForOp>(parent))
          if (isNspDistributedLoop(parentFor))
            return true;
      }
      return false;
    };

    // Helper for strip shard tensor annotations inside loops that were
    // explicitly distributed by this pass.
    //
    // Rationale:
    // In non-collective mode, the SPMD partitioning is expressed by the loop
    // schedule (lb/step adjusted using shard.process_linear_index). Any
    // shard.shard/shard.sharding inside such loops becomes stale metadata and
    // may confuse later lowerings (e.g., ShardToLLVM).
    //
    // We keep shard.grid and shard.process_linear_index, and we do NOT touch
    // shard annotations outside distributed loops.
    auto stripShardAnnotationsInDistributedLoops =
        [&](mlir::func::FuncOp func) {
          SmallVector<Operation *> eraseList;

          func.walk([&](mlir::scf::ForOp forOp) {
            if (!isNspDistributedLoop(forOp))
              return;

            Block *body = forOp.getBody();

            // Replace and erase shard.shard wrappers inside the loop body.
            body->walk([&](mlir::shard::ShardOp op) {
              // shard.shard is a value wrapper: replace result with input.
              if (op->getNumOperands() < 1 || op->getNumResults() < 1)
                return;
              Value wrapped = op->getOperand(0);
              Value res = op->getResult(0);
              res.replaceAllUsesWith(wrapped);
              eraseList.push_back(op);
            });

            // Erase shard.sharding descriptors inside the loop body if
            // unused. These usually become dead after removing shard.shard
            // wrappers.
            body->walk([&](mlir::shard::ShardingOp op) {
              if (op->getNumResults() != 1)
                return;
              if (op->getResult(0).use_empty())
                eraseList.push_back(op);
            });
          });

          for (Operation *op : eraseList)
            op->erase();
        };

    // Helper for Loop distribution (no collectives).
    //
    // For kernels with DOALL loops, sharding the reduced dimension requires
    // cross-shard reductions. In non-collective mode, the correct strategy is
    // to distribute the OUTER loop (e.g. batch) across NSPs and keep inner
    // vectors intact.
    //
    // We implement a simple cyclic distribution (block dist is a future work):
    //   original: for i = lb .. ub step s
    //   shard k : for i = lb + k*s .. ub step (s*numShards)
    //
    // This avoids any communication and preserves semantics for reductions
    // within the loop body.

    auto distributeScfForCyclic =
        [&](mlir::func::FuncOp func) -> LogicalResult {
      OpBuilder b(func.getContext());
      SmallVector<mlir::scf::ForOp> loops;
      func.walk([&](mlir::scf::ForOp forOp) { loops.push_back(forOp); });

      // Returns true iff `v` is derived from `iv` via a restricted set of
      // affine-like integer/index ops. This is a conservative "taint" analysis
      // used to detect whether the loop IV participates in output indexing.
      auto reachesFromIv = [&](Value iv, Value v) -> bool {
        if (v == iv)
          return true;

        // BFS over the def-use chain backwards (operand -> defining op -> its
        // operands).
        llvm::SmallVector<Value, 16> worklist;
        llvm::SmallPtrSet<Value, 32> visited;
        worklist.push_back(v);

        auto enqueue = [&](Value x) {
          if (!x)
            return;
          if (visited.insert(x).second)
            worklist.push_back(x);
        };

        while (!worklist.empty()) {
          Value cur = worklist.pop_back_val();
          if (cur == iv)
            return true;

          Operation *def = cur.getDefiningOp();
          if (!def)
            continue;

          // Allow common integer/index plumbing ops.
          // This intentionally ignores complex control/dataflow.
          if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp,
                  arith::DivUIOp, arith::RemSIOp, arith::RemUIOp, arith::ShLIOp,
                  arith::ShRSIOp, arith::ShRUIOp, arith::AndIOp, arith::OrIOp,
                  arith::XOrIOp, arith::ExtSIOp, arith::ExtUIOp,
                  arith::TruncIOp, arith::IndexCastOp, arith::IndexCastUIOp,
                  arith::ConstantOp>(def)) {
            for (Value opnd : def->getOperands())
              enqueue(opnd);
            continue;
          }

          // Allow memref.cast / view-like ops in the index path.
          if (isa<memref::CastOp>(def)) {
            for (Value opnd : def->getOperands())
              enqueue(opnd);
            continue;
          }
        }

        return false;
      };

      auto ofrReachesFromIv = [&](Value iv, OpFoldResult ofr) -> bool {
        if (auto v = dyn_cast<Value>(ofr))
          return reachesFromIv(iv, v);
        return false;
      };

      // Returns true iff the loop appears to perform per-iteration output
      // writes whose address depends on the IV, i.e. a DOALL-style tiled store.
      auto shouldDistributeLoop = [&](mlir::scf::ForOp forOp) -> bool {
        if (isNspDistributedLoop(forOp) ||
            hasNspDistributedLoopAncestor(forOp.getOperation()))
          return false;

        Value iv = forOp.getInductionVar();

        bool foundIvIndexedStore = false;

        // Direct memref.store with IV-derived indices.
        forOp.walk([&](memref::StoreOp st) {
          for (Value idx : st.getIndices()) {
            if (reachesFromIv(iv, idx)) {
              foundIvIndexedStore = true;
              return WalkResult::interrupt();
            }
          }
          return WalkResult::advance();
        });
        if (foundIvIndexedStore)
          return true;

        // Direct affine.store with IV-derived map operands. LayerNorm-like
        // kernels commonly store per-row side results such as mean/rstd this
        // way. These stores are safe for outer-loop distribution when their
        // address depends on the distributed IV.
        forOp.walk([&](affine::AffineStoreOp st) {
          for (Value idx : st.getMapOperands()) {
            if (reachesFromIv(iv, idx)) {
              foundIvIndexedStore = true;
              return WalkResult::interrupt();
            }
          }
          return WalkResult::advance();
        });
        if (foundIvIndexedStore)
          return true;

        // bufferization.materialize_in_destination where dest is a view into
        // the output buffer whose dynamic offset is derived from the IV. This
        // matches patterns like:
        //   %sv = memref.subview %dst[%off] ...
        //   bufferization.materialize_in_destination %t, %sv
        // as well as:
        //   %rc = memref.reinterpret_cast %dst to offset: [%off], sizes: [...]
        //   bufferization.materialize_in_destination %t, %rc
        forOp.walk([&](bufferization::MaterializeInDestinationOp mat) {
          Value dest = mat->getOperand(1);

          // Strip trivial casts.
          while (auto c = dest.getDefiningOp<memref::CastOp>())
            dest = c.getSource();

          // Case 1: memref.subview with dynamic offsets.
          if (auto sub = dest.getDefiningOp<memref::SubViewOp>()) {
            for (OpFoldResult ofr : sub.getMixedOffsets()) {
              if (ofrReachesFromIv(iv, ofr)) {
                foundIvIndexedStore = true;
                return WalkResult::interrupt();
              }
            }
            return WalkResult::advance();
          }

          // Case 2: memref.reinterpret_cast with dynamic offset/sizes/strides.
          if (auto rc = dest.getDefiningOp<memref::ReinterpretCastOp>()) {
            // Operands: source, offset, sizes..., strides...
            // Conservatively check all dynamic operands except the source.
            for (unsigned oi = 1, oe = rc->getNumOperands(); oi < oe; ++oi) {
              if (reachesFromIv(iv, rc->getOperand(oi))) {
                foundIvIndexedStore = true;
                return WalkResult::interrupt();
              }
            }
          }

          return WalkResult::advance();
        });

        return foundIvIndexedStore;
      };

      for (mlir::scf::ForOp forOp : loops) {
        // Bring-up constraints:
        //  - no iter_args / no results
        //  - induction var is index or an integer type with sufficient range
        if (!forOp.getResults().empty() || !forOp.getInitArgs().empty())
          continue;

        // Only distribute loops that look DOALL and "tile-store" safe:
        // the IV must participate in the output addressing (store/subview
        // offset).
        if (!shouldDistributeLoop(forOp))
          continue;

        mlir::Type ivTy = forOp.getInductionVar().getType();
        bool ivIsIndex = mlir::isa<mlir::IndexType>(ivTy);
        auto ivIntTy = mlir::dyn_cast<mlir::IntegerType>(ivTy);

        if (!ivIsIndex && !ivIntTy)
          continue;

        if (ivIntTy && ivIntTy.getWidth() < 32) {
          // Avoid silent overflow for small index types (e.g., i16).
          forOp.emitRemark()
              << "NSPLocalize: skipping scf.for distribution for "
                 "small integer IV type "
              << ivTy;
          continue;
        }

        Value lb = forOp.getLowerBound();
        Value ub = forOp.getUpperBound();
        Value step = forOp.getStep();

        // Require bounds/step to match the IV type to keep the transformation
        // simple.
        if (lb.getType() != ivTy || ub.getType() != ivTy ||
            step.getType() != ivTy)
          continue;

        Location loc = forOp.getLoc();
        b.setInsertionPoint(forOp);

        // procIdx: index in [0, numShards)
        Value procIdx = mlir::shard::ProcessLinearIndexOp::create(b, loc, grid);
        // Cast procIdx to the IV type (index stays index; integer gets
        // index_cast).
        Value procInIvTy = procIdx;
        if (!ivIsIndex)
          procInIvTy = arith::IndexCastOp::create(b, loc, ivTy, procIdx);

        // newLb = lb + procI32 * step
        Value offset = arith::MulIOp::create(b, loc, procInIvTy, step);
        Value newLb = arith::AddIOp::create(b, loc, lb, offset);

        // newStep = step * numShards
        Value cNum =
            ivIsIndex ? static_cast<Value>(
                            arith::ConstantIndexOp::create(b, loc, numShards))
                      : static_cast<Value>(arith::ConstantIntOp::create(
                            b, loc, numShards, ivIntTy.getWidth()));

        Value newStep = arith::MulIOp::create(b, loc, step, cNum);

        // Create the distributed loop.
        auto newFor = mlir::scf::ForOp::create(b, loc, newLb, ub, newStep);
        // Mark this loop so we can precisely clean up stale shard annotations
        // only within distributed loops (non-collective mode).
        newFor->setAttr("nsp.distribution", b.getStringAttr("cyclic"));

        // Clone the body operations, remapping the induction variable.
        Block *oldBody = forOp.getBody();
        Block *newBody = newFor.getBody();

        // Remove the default terminator inserted by builder.
        newBody->getOperations().clear();

        IRMapping mapping;
        mapping.map(forOp.getInductionVar(), newFor.getInductionVar());

        for (Operation &op : oldBody->without_terminator()) {
          b.setInsertionPointToEnd(newBody);
          b.clone(op, mapping);
        }
        b.setInsertionPointToEnd(newBody);
        mlir::scf::YieldOp::create(b, loc);

        // Replace uses of the old loop results (none here) and erase old loop.
        forOp.erase();
      }
      return success();
    };

    if (!allowCollectives) {
      // In non-collective mode, prioritize loop distribution. This is the
      // required strategy for patterns with intra-tile reductions (e.g.
      // softmax).
      module.walk([&](mlir::func::FuncOp func) {
        if (failed(distributeScfForCyclic(func))) {
          signalPassFailure();
          return;
        }
      });

      // After distribution, strip shard tensor annotations only inside
      // distributed loops. Outside those loops, sharding metadata (if any)
      // is preserved.
      module.walk([&](mlir::func::FuncOp func) {
        stripShardAnnotationsInDistributedLoops(func);
      });
    }

    // Optionally sanity-check that we have some sharding descriptors.
    // This is a warning (not a hard error) because some pipelines may
    // legally run with no sharding yet (e.g. early bring-up).
    int64_t numShardingOps = 0;
    module.walk([&](mlir::shard::ShardingOp op) { ++numShardingOps; });

    if (numShardingOps == 0) {
      module.emitWarning()
          << "NSPLocalizePass found shard.grid '@nsp' but no 'shard.sharding' "
             "ops in the module. This may be expected during early bring-up, "
             "but if it is unexpected, ensure sharding propagation created "
             "sharding descriptors.";
    }

    // Minimal bring-up localization.
    //
    // We intentionally keep this scoped:
    //   - Only elementwise linalg.generic with identity indexing maps.
    //   - Identity-map elementwise tensors up to rank 4.
    //   - The existing rank-2 broadcast path remains specialized.
    //   - Only all-parallel iterator spaces.
    //   - Only a 1-D NSP grid; the tensor split axis is chosen per op.
    //   - Only ranked tensors with at least one positive static dimension
    //     divisible by the grid size.
    const SmallVector<mlir::shard::GridAxis> gridAxes = {
        static_cast<mlir::shard::GridAxis>(0)};
    // Some Shard ops (e.g. all_gather) expect grid_axes as i16.
    const SmallVector<int16_t> gridAxesI16 = {static_cast<int16_t>(0)};

    int64_t splitAxis = 0;

    auto isValidSplitAxis = [&](RankedTensorType globalTy,
                                int64_t axis) -> bool {
      if (!globalTy || axis < 0 || axis >= globalTy.getRank())
        return false;
      int64_t dim = globalTy.getDimSize(axis);
      return !ShapedType::isDynamic(dim) && dim > 0 && dim % numShards == 0;
    };

    // Prefer axis 0 to preserve the existing behavior, but allow higher-rank
    // block_ptr tensors with a leading unit dimension (e.g. 1x512x512x4) to be
    // sharded along the first static dimension that can be evenly partitioned.
    auto chooseSplitAxis =
        [&](RankedTensorType globalTy) -> std::optional<int64_t> {
      if (!globalTy || globalTy.getRank() == 0)
        return std::nullopt;
      if (isValidSplitAxis(globalTy, 0))
        return 0;
      for (int64_t axis = 1; axis < globalTy.getRank(); ++axis)
        if (isValidSplitAxis(globalTy, axis))
          return axis;
      return std::nullopt;
    };

    // Helper to compute a "local" tensor type for the current split axis.
    auto getLocalType = [&](RankedTensorType globalTy) -> RankedTensorType {
      if (!globalTy || globalTy.getRank() == 0 || splitAxis < 0 ||
          splitAxis >= globalTy.getRank())
        return RankedTensorType();
      int64_t splitDim = globalTy.getDimSize(splitAxis);
      if (ShapedType::isDynamic(splitDim))
        return RankedTensorType();
      if (splitDim % numShards != 0)
        return RankedTensorType();
      SmallVector<int64_t> newShape(globalTy.getShape().begin(),
                                    globalTy.getShape().end());
      newShape[splitAxis] = splitDim / numShards;
      return RankedTensorType::get(newShape, globalTy.getElementType(),
                                   globalTy.getEncoding());
    };

    // Return the shard-local shape obtained by splitting along axis 0.
    // This intentionally ignores element type and encoding so that
    // type-changing elementwise ops (e.g. f16 -> f32) can still be
    // recognized as shard-compatible.
    auto getLocalShape =
        [&](RankedTensorType globalTy) -> std::optional<SmallVector<int64_t>> {
      auto localTy = getLocalType(globalTy);
      if (!localTy)
        return std::nullopt;
      return SmallVector<int64_t>(localTy.getShape().begin(),
                                  localTy.getShape().end());
    };

    auto haveSameLocalShape = [&](RankedTensorType a,
                                  RankedTensorType b) -> bool {
      auto aShape = getLocalShape(a);
      auto bShape = getLocalShape(b);
      return aShape && bShape && (*aShape == *bShape);
    };

    auto isCompatibleInputForBroadcastLocalize =
        [&](RankedTensorType inTy, AffineMap inputMap,
            RankedTensorType outTy) -> bool {
      if (!inTy || !outTy || outTy.getRank() != 2)
        return false;

      auto outLocalTy = getLocalType(outTy);
      if (!outLocalTy)
        return false;

      if (inTy.getRank() == 2) {
        if (!isIdentity2DMap(inputMap))
          return false;
        auto inLocalTy = getLocalType(inTy);
        if (!inLocalTy)
          return false;
        return inLocalTy.getShape() == outLocalTy.getShape();
      }

      if (inTy.getRank() == 1 && isRowBroadcastMap(inputMap)) {
        auto inLocalTy = getLocalType(inTy);
        if (!inLocalTy)
          return false;
        return inLocalTy.getShape().size() == 1 &&
               outLocalTy.getShape().size() == 2 &&
               inLocalTy.getShape()[0] == outLocalTy.getShape()[0];
      }

      if (inTy.getRank() == 1 && isColumnBroadcastMap(inputMap)) {
        if (outLocalTy.getShape().size() != 2)
          return false;
        int64_t colExtent = inTy.getDimSize(0);
        int64_t localCols = outLocalTy.getShape()[1];
        if (ShapedType::isDynamic(colExtent) ||
            ShapedType::isDynamic(localCols))
          return false;
        return colExtent == localCols;
      }

      return false;
    };

    // Compute the shard-local tensor type expected for a specific operand of
    // a localized generic.
    //
    // For the current broadcast-aware path:
    //   - rank-2 identity inputs use the same local type as the result,
    //   - rank-1 row-broadcast inputs ((d0, d1) -> (d0)) use only the local
    //     row extent, and
    //   - rank-1 column-broadcast inputs ((d0, d1) -> (d1)) are replicated and
    //     keep the full column extent.
    auto getExpectedLocalTypeForOperand =
        [&](RankedTensorType operandTy, AffineMap operandMap,
            RankedTensorType localResultTy) -> RankedTensorType {
      if (!operandTy || !localResultTy)
        return RankedTensorType();

      if (operandTy.getRank() == 2 && isIdentity2DMap(operandMap))
        return localResultTy;

      if (operandTy.getRank() == 1 && isRowBroadcastMap(operandMap)) {
        if (localResultTy.getRank() != 2)
          return RankedTensorType();

        int64_t localRows = localResultTy.getShape()[0];
        if (ShapedType::isDynamic(localRows))
          return RankedTensorType();

        return RankedTensorType::get({localRows}, operandTy.getElementType(),
                                     operandTy.getEncoding());
      }

      if (operandTy.getRank() == 1 && isColumnBroadcastMap(operandMap)) {
        if (localResultTy.getRank() != 2)
          return RankedTensorType();

        int64_t localCols = localResultTy.getShape()[1];
        int64_t operandCols = operandTy.getDimSize(0);
        if (ShapedType::isDynamic(localCols) ||
            ShapedType::isDynamic(operandCols) || localCols != operandCols)
          return RankedTensorType();

        return RankedTensorType::get({localCols}, operandTy.getElementType(),
                                     operandTy.getEncoding());
      }

      return RankedTensorType();
    };

    auto isSupportedElementwiseGenericShape =
        [&](mlir::linalg::GenericOp g, RankedTensorType outResTy,
            ArrayRef<RankedTensorType> inputTys) -> bool {
      if (!outResTy)
        return false;

      const int64_t rank = outResTy.getRank();
      if (!isSupportedIdentityElementwiseRank(rank))
        return false;

      if (g.getNumLoops() != rank)
        return false;

      auto iters = g.getIteratorTypesArray();
      if ((int64_t)iters.size() != rank)
        return false;
      if (!llvm::all_of(iters, [](mlir::utils::IteratorType it) {
            return it == mlir::utils::IteratorType::parallel;
          }))
        return false;

      auto maps = g.getIndexingMapsArray();
      if ((int64_t)maps.size() != g.getNumDpsInputs() + g.getNumDpsInits())
        return false;
      if (!llvm::all_of(maps, [](AffineMap m) { return m.isIdentity(); }))
        return false;

      for (RankedTensorType t : inputTys) {
        if (!t || t.getRank() != rank)
          return false;
      }

      return true;
    };

    auto hasCompatibleRegionArgumentTypes = [](Block &block, ValueRange inputs,
                                               ValueRange outputs) -> bool {
      if (block.getNumArguments() != inputs.size() + outputs.size())
        return false;

      auto getElementType = [](Type t) -> Type {
        if (auto shaped = dyn_cast<ShapedType>(t))
          return shaped.getElementType();
        return t;
      };

      unsigned argIdx = 0;
      for (Value input : inputs) {
        if (block.getArgument(argIdx++).getType() !=
            getElementType(input.getType()))
          return false;
      }

      for (Value output : outputs) {
        if (block.getArgument(argIdx++).getType() !=
            getElementType(output.getType()))
          return false;
      }

      return true;
    };

    auto cloneLinalgRegion = [&](Region &srcRegion, Region &dstRegion) {
      dstRegion.getBlocks().clear();

      Block &srcBlock = srcRegion.front();
      Block *dstBlock = new Block();
      dstRegion.push_back(dstBlock);

      for (BlockArgument arg : srcBlock.getArguments())
        dstBlock->addArgument(arg.getType(), arg.getLoc());

      IRMapping map;
      for (auto [srcArg, dstArg] :
           llvm::zip(srcBlock.getArguments(), dstBlock->getArguments()))
        map.map(srcArg, dstArg);

      OpBuilder nb = OpBuilder::atBlockEnd(dstBlock);
      for (Operation &op : srcBlock.getOperations())
        nb.clone(op, map);
    };

    auto getLocalTypeWithShape =
        [&](RankedTensorType globalTy,
            ArrayRef<int64_t> localShape) -> RankedTensorType {
      if (!globalTy || (int64_t)localShape.size() != globalTy.getRank())
        return RankedTensorType();
      for (int64_t dim : localShape)
        if (ShapedType::isDynamic(dim) || dim <= 0)
          return RankedTensorType();
      return RankedTensorType::get(localShape, globalTy.getElementType(),
                                   globalTy.getEncoding());
    };

    auto isShapePreservingElementwiseGeneric =
        [&](mlir::linalg::GenericOp prod,
            RankedTensorType expectedLocalTy) -> bool {
      if (!expectedLocalTy)
        return false;

      const int64_t rank = expectedLocalTy.getRank();
      if (!isSupportedIdentityElementwiseRank(rank))
        return false;

      const int64_t nIn = prod.getNumDpsInputs();
      if (nIn < 1 || nIn > 3 || prod.getNumDpsInits() != 1)
        return false;

      if (prod.getNumLoops() != rank)
        return false;

      auto iters = prod.getIteratorTypesArray();
      if ((int64_t)iters.size() != rank)
        return false;
      if (!llvm::all_of(iters, [](mlir::utils::IteratorType it) {
            return it == mlir::utils::IteratorType::parallel;
          }))
        return false;

      auto maps = prod.getIndexingMapsArray();
      if ((int64_t)maps.size() != nIn + 1)
        return false;
      if (!llvm::all_of(maps, [](AffineMap m) { return m.isIdentity(); }))
        return false;

      auto resTy = dyn_cast<RankedTensorType>(prod.getResult(0).getType());
      if (!resTy || resTy.getRank() != rank)
        return false;

      for (OpOperand *in : prod.getDpsInputOperands()) {
        auto inTy = dyn_cast<RankedTensorType>(in->get().getType());
        if (!inTy || inTy.getRank() != rank)
          return false;
      }

      return true;
    };

    auto getTransposeInputLocalType =
        [&](linalg::TransposeOp transpose,
            RankedTensorType expectedResultLocalTy) -> RankedTensorType {
      auto inputTy = dyn_cast<RankedTensorType>(transpose.getInput().getType());
      if (!inputTy || !expectedResultLocalTy)
        return RankedTensorType();

      auto perm = transpose.getPermutation();
      if ((int64_t)perm.size() != inputTy.getRank() ||
          inputTy.getRank() != expectedResultLocalTy.getRank())
        return RankedTensorType();

      SmallVector<int64_t> inputLocalShape(inputTy.getRank(),
                                           ShapedType::kDynamic);
      ArrayRef<int64_t> resultShape = expectedResultLocalTy.getShape();

      // linalg.transpose uses:
      //   result_dim[i] = input_dim[permutation[i]]
      // Therefore, if the localized result shape is known, the localized input
      // shape is obtained by applying the inverse mapping.
      for (auto [resultDim, inputDim] : llvm::enumerate(perm)) {
        if (inputDim < 0 || inputDim >= inputTy.getRank())
          return RankedTensorType();
        inputLocalShape[inputDim] = resultShape[resultDim];
      }

      for (int64_t dim : inputLocalShape)
        if (ShapedType::isDynamic(dim) || dim <= 0)
          return RankedTensorType();

      return RankedTensorType::get(inputLocalShape, inputTy.getElementType(),
                                   inputTy.getEncoding());
    };

    auto getReduceInputLocalType =
        [&](linalg::ReduceOp reduce,
            RankedTensorType expectedResultLocalTy) -> RankedTensorType {
      if (!expectedResultLocalTy || reduce.getInputs().size() != 1 ||
          reduce.getInits().size() != 1)
        return RankedTensorType();

      auto inputTy =
          dyn_cast<RankedTensorType>(reduce.getInputs()[0].getType());
      if (!inputTy)
        return RankedTensorType();

      llvm::SmallDenseSet<int64_t> reducedDims;
      for (int64_t dim : reduce.getDimensions())
        reducedDims.insert(dim);

      if ((int64_t)reducedDims.size() !=
          inputTy.getRank() - expectedResultLocalTy.getRank())
        return RankedTensorType();

      SmallVector<int64_t> inputLocalShape;
      inputLocalShape.reserve(inputTy.getRank());

      int64_t outDim = 0;
      for (int64_t dim = 0; dim < inputTy.getRank(); ++dim) {
        if (reducedDims.contains(dim)) {
          int64_t extent = inputTy.getDimSize(dim);
          if (ShapedType::isDynamic(extent) || extent <= 0)
            return RankedTensorType();
          inputLocalShape.push_back(extent);
          continue;
        }

        if (outDim >= expectedResultLocalTy.getRank())
          return RankedTensorType();
        inputLocalShape.push_back(expectedResultLocalTy.getDimSize(outDim++));
      }

      return RankedTensorType::get(inputLocalShape, inputTy.getElementType(),
                                   inputTy.getEncoding());
    };

    //===------------------------------------------------------------------===//
    // tensor.expand_shape localization helpers
    //===------------------------------------------------------------------===//

    // Return, for each source dimension of tensor.expand_shape, the result
    // dimension that should carry the source sharding/local extent.
    //
    // This only supports the unit-expansion case: tensor<M> -> tensor<M x 1>
    //
    // The original dimension maps to the only non-unit dimension in the
    // reassociation group. Newly inserted unit dimensions remain replicated.
    //
    // General reshapes are not supported here, because they are not simple
    // view-like sharding projections.
    auto getUnitExpandRepresentativeDims = [&](tensor::ExpandShapeOp expand)
        -> std::optional<SmallVector<int64_t>> {
      auto srcTy = dyn_cast<RankedTensorType>(expand->getOperand(0).getType());
      auto resultTy =
          dyn_cast<RankedTensorType>(expand->getResult(0).getType());
      if (!srcTy || !resultTy)
        return std::nullopt;

      SmallVector<ReassociationIndices> reassociation =
          expand.getReassociationIndices();

      if ((int64_t)reassociation.size() != srcTy.getRank())
        return std::nullopt;

      SmallVector<int64_t> representatives;
      representatives.reserve(srcTy.getRank());

      for (ArrayRef<int64_t> group : reassociation) {
        if (group.empty())
          return std::nullopt;

        int64_t representative = -1;
        int64_t nonUnitDims = 0;

        for (int64_t resultDim : group) {
          if (resultDim < 0 || resultDim >= resultTy.getRank())
            return std::nullopt;

          int64_t dimSize = resultTy.getDimSize(resultDim);

          // Dynamic dimensions inside a multi-dim group cannot be proven to be
          // unit-expansion. Keep this path conservative.
          if (ShapedType::isDynamic(dimSize)) {
            if (group.size() != 1)
              return std::nullopt;
            representative = resultDim;
            ++nonUnitDims;
            continue;
          }

          if (dimSize != 1) {
            representative = resultDim;
            ++nonUnitDims;
          }
        }

        if (nonUnitDims > 1)
          return std::nullopt;

        // Degenerate case: all expanded dims are unit dims.
        if (representative < 0)
          representative = group.front();

        representatives.push_back(representative);
      }

      return representatives;
    };

    auto getExpandShapeInputLocalType =
        [&](tensor::ExpandShapeOp expand,
            RankedTensorType expectedResultLocalTy) -> RankedTensorType {
      auto inputTy =
          dyn_cast<RankedTensorType>(expand->getOperand(0).getType());
      auto resultTy =
          dyn_cast<RankedTensorType>(expand->getResult(0).getType());
      if (!inputTy || !resultTy || !expectedResultLocalTy)
        return RankedTensorType();

      if (expectedResultLocalTy.getRank() != resultTy.getRank())
        return RankedTensorType();

      auto representatives = getUnitExpandRepresentativeDims(expand);
      if (!representatives ||
          (int64_t)representatives->size() != inputTy.getRank())
        return RankedTensorType();

      SmallVector<int64_t> inputLocalShape;
      inputLocalShape.reserve(inputTy.getRank());

      for (int64_t resultDim : *representatives) {
        int64_t localDim = expectedResultLocalTy.getDimSize(resultDim);
        if (ShapedType::isDynamic(localDim) || localDim <= 0)
          return RankedTensorType();
        inputLocalShape.push_back(localDim);
      }

      return RankedTensorType::get(inputLocalShape, inputTy.getElementType(),
                                   inputTy.getEncoding());
    };

    auto getExpandShapeLocalResultType =
        [&](tensor::ExpandShapeOp expand,
            RankedTensorType localInputTy) -> RankedTensorType {
      auto resultTy =
          dyn_cast<RankedTensorType>(expand->getResult(0).getType());
      if (!resultTy || !localInputTy)
        return RankedTensorType();

      auto representatives = getUnitExpandRepresentativeDims(expand);
      if (!representatives ||
          (int64_t)representatives->size() != localInputTy.getRank())
        return RankedTensorType();

      SmallVector<int64_t> localResultShape(resultTy.getShape().begin(),
                                            resultTy.getShape().end());

      for (auto [srcDim, resultDim] : llvm::enumerate(*representatives)) {
        int64_t localDim = localInputTy.getDimSize(srcDim);
        if (ShapedType::isDynamic(localDim) || localDim <= 0)
          return RankedTensorType();
        localResultShape[resultDim] = localDim;
      }

      return RankedTensorType::get(localResultShape, resultTy.getElementType(),
                                   resultTy.getEncoding());
    };

    module.walk([&](mlir::func::FuncOp func) {
      OpBuilder b(func.getContext());

      auto buildTensorEmptyLike = [&](Location loc,
                                      RankedTensorType ty) -> Value {
        return tensor::EmptyOp::create(b, loc, ty.getShape(),
                                       ty.getElementType());
      };

      using MaterializeSink =
          std::pair<bufferization::MaterializeInDestinationOp,
                    SmallVector<Operation *>>;

      // Find a bufferization.materialize_in_destination sink reachable from
      // `v` through a supported wrapper path.
      //
      // Sharding propagation often wraps values with shard.shard, so the IR can
      // look like:
      //   %r  = linalg.generic ... -> tensor<...>
      //   %r1 = shard.shard %r to %sharding
      //   bufferization.materialize_in_destination %r1 in %dst
      //
      // In producer/consumer chains, the wrapper result may have additional
      // users, for example:
      //   %r1 = shard.shard %r to %sharding
      //   %e  = tensor.expand_shape %r1 ...
      //   bufferization.materialize_in_destination %e in %dst
      //   %r2 = shard.shard %r1 to %other_sharding annotate_for_users
      //   %next = linalg.generic ins(%r2, ...)
      //
      // The materialization side path is still valid and should be localized.
      // Therefore this helper searches among the users instead of requiring
      // every value in the wrapper chain to be single-use.
      //
      // In store-by-tile mode, we rewrite the materialization to store the
      // local tile into a subview of %dst. Only now-dead wrappers are erased;
      // wrappers/producers that still feed downstream computation are kept.
      auto findMaterializeSink =
          [&](Value v) -> std::optional<MaterializeSink> {
        SmallVector<Operation *> wrappers;

        std::function<std::optional<MaterializeSink>(Value, unsigned)> search =
            [&](Value cur, unsigned depth) -> std::optional<MaterializeSink> {
          if (depth > 8)
            return std::nullopt;

          SmallVector<Operation *> users;
          for (Operation *user : cur.getUsers())
            users.push_back(user);

          // Prefer a direct materialization sink when present.
          for (Operation *user : users) {
            if (auto mat =
                    dyn_cast<bufferization::MaterializeInDestinationOp>(user)) {
              if (mat->getOperand(0) != cur)
                continue;
              return std::make_optional(std::make_pair(mat, wrappers));
            }
          }

          // Otherwise search through supported tensor/shard wrappers.
          for (Operation *user : users) {
            // Allow shard.shard wrappers.
            if (auto shardOp = dyn_cast<mlir::shard::ShardOp>(user)) {
              if (shardOp->getNumOperands() < 1 ||
                  shardOp->getOperand(0) != cur)
                continue;

              wrappers.push_back(user);
              auto found = search(shardOp->getResult(0), depth + 1);
              if (found)
                return found;
              wrappers.pop_back();
              continue;
            }

            // Allow an optional tensor.cast in between.
            if (auto castOp = dyn_cast<tensor::CastOp>(user)) {
              if (castOp.getSource() != cur)
                continue;
              wrappers.push_back(user);
              auto found = search(castOp.getResult(), depth + 1);
              if (found)
                return found;
              wrappers.pop_back();
              continue;
            }

            // Allow a unit-expanding tensor.expand_shape in the final sink
            // chain. Example:
            //   %r = linalg.generic ... : tensor<Mxf32>
            //   %e = tensor.expand_shape %r [[0, 1]]
            //          output_shape [M, 1]
            //          : tensor<Mxf32> into tensor<Mx1xf32>
            //   bufferization.materialize_in_destination %e in %dst
            //
            // In store-by-tile mode, the localized chain must become:
            //   %r.local = ...
            //   %e.local = tensor.expand_shape %r.local [[0, 1]]
            //                : tensor<tileMxf32> into tensor<tileMx1xf32>
            if (auto expandOp = dyn_cast<tensor::ExpandShapeOp>(user)) {
              if (expandOp->getOperand(0) != cur)
                continue;
              if (!getUnitExpandRepresentativeDims(expandOp))
                continue;

              wrappers.push_back(user);
              auto found = search(expandOp->getResult(0), depth + 1);
              if (found)
                return found;
              wrappers.pop_back();
              continue;
            }
          }

          return std::nullopt;
        };

        return search(v, /*depth=*/0);
      };

      // Strip trivial wrapper ops (shard.shard, tensor.cast) to expose a real
      // defining op, that is, wrappers that may sit between a tensor value
      // and its backing buffer.
      auto stripTrivialWrappers = [&](Value v) -> Value {
        Value cur = v;
        while (Operation *def = cur.getDefiningOp()) {
          if (auto shardOp = dyn_cast<mlir::shard::ShardOp>(def)) {
            if (shardOp->getNumOperands() < 1)
              break;
            cur = shardOp->getOperand(0);
            continue;
          }
          if (auto castOp = dyn_cast<tensor::CastOp>(def)) {
            cur = castOp.getSource();
            continue;
          }
          break;
        }
        return cur;
      };

      struct TransposeReduceChainMatch {
        mlir::linalg::TransposeOp transpose;
        mlir::linalg::GenericOp unaryGeneric;
      };

      auto isUnaryRank2IdentityGeneric =
          [&](mlir::linalg::GenericOp generic) -> bool {
        if (!generic)
          return false;

        if (generic.getNumDpsInputs() != 1 || generic.getNumDpsInits() != 1)
          return false;

        if (generic.getNumLoops() != 2)
          return false;

        auto iters = generic.getIteratorTypesArray();
        if (iters.size() != 2)
          return false;
        if (!llvm::all_of(iters, [](mlir::utils::IteratorType it) {
              return it == mlir::utils::IteratorType::parallel;
            }))
          return false;

        auto maps = generic.getIndexingMapsArray();
        if (maps.size() != 2)
          return false;
        if (!isIdentity2DMap(maps[0]) || !isIdentity2DMap(maps[1]))
          return false;

        auto inputTy = dyn_cast<RankedTensorType>(
            generic.getDpsInputOperand(0)->get().getType());
        auto resultTy =
            dyn_cast<RankedTensorType>(generic.getResult(0).getType());
        if (!inputTy || !resultTy)
          return false;

        if (inputTy.getRank() != 2 || resultTy.getRank() != 2)
          return false;

        if (inputTy.getShape() != resultTy.getShape())
          return false;

        return true;
      };

      auto matchTransposeReduceChain =
          [&](mlir::linalg::ReduceOp reduce,
              TransposeReduceChainMatch &match) -> bool {
        if (!reduce || reduce.getInputs().size() != 1 ||
            reduce.getInits().size() != 1)
          return false;

        auto reduceResultTy =
            dyn_cast<RankedTensorType>(reduce->getResult(0).getType());
        if (!reduceResultTy || reduceResultTy.getRank() != 1)
          return false;

        auto dims = reduce.getDimensions();
        if (dims.size() != 1 || dims[0] != 0)
          return false;

        Value reduceInput = stripTrivialWrappers(reduce.getInputs()[0]);

        // Common blocked-softmax form:
        //   transpose -> unary identity generic, usually extf -> reduce(dim 0)
        if (auto generic =
                reduceInput.getDefiningOp<mlir::linalg::GenericOp>()) {
          if (!isUnaryRank2IdentityGeneric(generic))
            return false;

          Value genericInput =
              stripTrivialWrappers(generic.getDpsInputOperand(0)->get());
          auto transpose =
              genericInput.getDefiningOp<mlir::linalg::TransposeOp>();
          if (!transpose)
            return false;

          match.transpose = transpose;
          match.unaryGeneric = generic;
        } else {
          // Simpler form:
          //   transpose -> reduce(dim 0)
          auto transpose =
              reduceInput.getDefiningOp<mlir::linalg::TransposeOp>();
          if (!transpose)
            return false;

          match.transpose = transpose;
          match.unaryGeneric = mlir::linalg::GenericOp();
        }

        auto inputTy =
            dyn_cast<RankedTensorType>(match.transpose.getInput().getType());
        auto transposeResultTy =
            dyn_cast<RankedTensorType>(match.transpose->getResult(0).getType());
        if (!inputTy || !transposeResultTy)
          return false;

        if (inputTy.getRank() != 2 || transposeResultTy.getRank() != 2)
          return false;

        auto perm = match.transpose.getPermutation();
        if (perm.size() != 2 || perm[0] != 1 || perm[1] != 0)
          return false;

        return true;
      };

      auto cloneUnaryGenericOnLocalInput = [&](mlir::linalg::GenericOp generic,
                                               Value localInput) -> Value {
        if (!generic || !localInput)
          return Value();

        auto inputTy = dyn_cast<RankedTensorType>(localInput.getType());
        auto oldResultTy =
            dyn_cast<RankedTensorType>(generic.getResult(0).getType());
        if (!inputTy || !oldResultTy)
          return Value();

        RankedTensorType localResultTy = RankedTensorType::get(
            inputTy.getShape(), oldResultTy.getElementType(),
            oldResultTy.getEncoding());

        Value localInit = tensor::EmptyOp::create(
            b, generic.getLoc(), localResultTy.getShape(),
            localResultTy.getElementType());

        auto localGeneric = mlir::linalg::GenericOp::create(
            b, generic.getLoc(),
            /*resultTensorTypes=*/TypeRange{localResultTy},
            /*inputs=*/ValueRange{localInput},
            /*outputs=*/ValueRange{localInit},
            /*indexingMaps=*/generic.getIndexingMapsArray(),
            /*iteratorTypes=*/generic.getIteratorTypesArray(),
            /*doc=*/StringRef(), /*libraryCall=*/StringRef());

        if (!hasCompatibleRegionArgumentTypes(generic.getRegion().front(),
                                              ValueRange{localInput},
                                              ValueRange{localInit}))
          return Value();

        cloneLinalgRegion(generic.getRegion(), localGeneric.getRegion());
        return localGeneric.getResult(0);
      };

      auto isCompatibleElementwiseGeneric =
          [&](mlir::linalg::GenericOp prod,
              RankedTensorType expectedLocalTy) -> bool {
        if (!expectedLocalTy)
          return false;

        const int64_t expectedRank = expectedLocalTy.getRank();
        if (!isSupportedIdentityElementwiseRank(expectedRank))
          return false;

        const int64_t nIn = prod.getNumDpsInputs();
        if ((nIn != 1 && nIn != 2) || prod.getNumDpsInits() != 1)
          return false;

        if (prod.getNumLoops() != expectedRank)
          return false;

        auto iters = prod.getIteratorTypesArray();
        if ((int64_t)iters.size() != expectedRank)
          return false;
        if (!llvm::all_of(iters, [](mlir::utils::IteratorType it) {
              return it == mlir::utils::IteratorType::parallel;
            }))
          return false;

        auto maps = prod.getIndexingMapsArray();
        if ((int64_t)maps.size() != nIn + 1)
          return false;
        for (AffineMap m : maps)
          if (!m.isIdentity())
            return false;

        auto resTy = dyn_cast<RankedTensorType>(prod.getResult(0).getType());
        if (!resTy || resTy.getRank() != expectedRank)
          return false;

        auto prodLocalTy = getLocalType(resTy);
        if (!prodLocalTy)
          return false;

        if (prodLocalTy.getShape() != expectedLocalTy.getShape())
          return false;

        for (int64_t i = 0; i < nIn; ++i) {
          auto inTy = dyn_cast<RankedTensorType>(
              prod.getDpsInputOperand(i)->get().getType());
          if (!inTy || inTy.getRank() != expectedRank)
            return false;

          auto inLocalTy = getLocalType(inTy);
          if (!inLocalTy)
            return false;

          if (inLocalTy.getShape() != expectedLocalTy.getShape())
            return false;
        }

        return true;
      };

      llvm::DenseMap<Operation *, mlir::scf::ForOp> localizedScfForCache;
      llvm::SmallPtrSet<Operation *, 16> failedLocalizedScfForCache;

      std::function<LogicalResult(mlir::scf::ForOp,
                                  llvm::DenseMap<Value, Value> &)>
          localizeScfForWithTensorIterArgs;

      // Build a local-tile version of a global value by cloning a chain of
      // compatible producers. This avoids materializing full global
      // intermediates when only the final store is tiled.
      std::function<Value(Value, RankedTensorType,
                          llvm::DenseMap<Value, Value> &)>
          materializeLocalValue =
              [&](Value v, RankedTensorType expectedLocalTy,
                  llvm::DenseMap<Value, Value> &cache) -> Value {
        Value base = stripTrivialWrappers(v);
        auto it = cache.find(base);
        if (it != cache.end())
          return it->second;

        // Replicated values do not need slicing. This is required for
        // column-wise broadcast operands such as layer-norm gamma/beta:
        //   tensor<N>, map (d0, d1) -> (d1)
        // where N is the full column extent and must remain available on every
        // NSP participant.
        auto baseTy = dyn_cast<RankedTensorType>(base.getType());
        if (baseTy && baseTy == expectedLocalTy) {
          cache[base] = base;
          return base;
        }

        // linalg.fill: produce a local constant tile.
        if (auto fill =
                dyn_cast_or_null<mlir::linalg::FillOp>(base.getDefiningOp())) {
          auto fillResTy = dyn_cast<RankedTensorType>(base.getType());
          if (!fillResTy)
            return Value();

          RankedTensorType fillLocalTy =
              getLocalTypeWithShape(fillResTy, expectedLocalTy.getShape());
          if (!fillLocalTy)
            return Value();

          Value outLocalInit = mlir::tensor::EmptyOp::create(
              b, fill.getLoc(), fillLocalTy.getShape(),
              fillLocalTy.getElementType());
          auto localFill = mlir::linalg::FillOp::create(
              b, fill.getLoc(), /*inputs=*/fill.getInputs(),
              /*outputs=*/ValueRange{outLocalInit});
          Value localRes = localFill.getResult(0);
          cache[base] = localRes;
          return localRes;
        }

        // linalg.transpose: localize the source and rebuild the transpose on
        // the local tile. This is important for row-wise softmax patterns:
        //
        //   input[MxN] -> transpose[NxM] -> reduce(dim 0) -> tensor<M>
        //
        // If the outer/original M dimension is sharded, the transpose-local
        // shape becomes [N x localM], not [localN x M]. Rebuilding the
        // transpose locally avoids slicing the transposed global tensor along
        // the wrong physical dimension.
        if (auto transpose = dyn_cast_or_null<mlir::linalg::TransposeOp>(
                base.getDefiningOp())) {
          RankedTensorType inputLocalTy =
              getTransposeInputLocalType(transpose, expectedLocalTy);
          if (inputLocalTy) {
            Value localInput = materializeLocalValue(transpose.getInput(),
                                                     inputLocalTy, cache);
            if (localInput) {
              Value outLocalInit =
                  buildTensorEmptyLike(transpose.getLoc(), expectedLocalTy);
              auto localTranspose = mlir::linalg::TransposeOp::create(
                  b, transpose.getLoc(),
                  /*input=*/localInput,
                  /*init=*/outLocalInit,
                  /*permutation=*/transpose.getPermutation());

              Value localRes = localTranspose.getResult()[0];
              cache[base] = localRes;
              return localRes;
            }
          }
        }

        // linalg.reduce: localize reductions when the sharded dimension is a
        // non-reduced dimension. This is the non-collective softmax case:
        // each NSP owns complete rows, while the column reduction remains
        // fully local to that row.
        if (auto reduce = dyn_cast_or_null<mlir::linalg::ReduceOp>(
                base.getDefiningOp())) {

          // Optimize the canonicalized row-wise softmax reduction form.
          //
          // Some softmax variants reach NSPSharding as:
          //   input<M x N>
          //     -> transpose [1, 0]
          //     -> optional unary identity generic, usually f16 -> f32
          //     -> reduce(dim 0)
          //     -> tensor<M>
          //
          // The localized fallback would rebuild:
          //   input<localM x N>
          //     -> transpose<N x localM>
          //     -> optional unary generic
          //     -> reduce(dim 0)
          //
          // Rebuild this directly as:
          //   input<localM x N>
          //     -> optional unary generic
          //     -> reduce(dim 1)
          //
          // This preserves the row-wise reduction semantics and avoids the
          // local transpose tensor and transpose operation.
          TransposeReduceChainMatch chain;
          if (matchTransposeReduceChain(reduce, chain)) {
            RankedTensorType oldReduceInputLocalTy =
                getReduceInputLocalType(reduce, expectedLocalTy);
            RankedTensorType sourceLocalTy = getTransposeInputLocalType(
                chain.transpose, oldReduceInputLocalTy);

            if (sourceLocalTy) {
              Value localSource = materializeLocalValue(
                  chain.transpose.getInput(), sourceLocalTy, cache);

              if (localSource) {
                Value localReduceInput = localSource;

                if (chain.unaryGeneric) {
                  localReduceInput = cloneUnaryGenericOnLocalInput(
                      chain.unaryGeneric, localSource);
                }

                Value localInit = materializeLocalValue(reduce.getInits()[0],
                                                        expectedLocalTy, cache);

                if (localReduceInput && localInit) {
                  auto inputTy =
                      dyn_cast<RankedTensorType>(localReduceInput.getType());
                  if (!inputTy)
                    return Value();

                  // The matched transpose is [1, 0], and the original reduce
                  // dimension is 0 in the transposed tensor. Therefore the
                  // equivalent reduction dimension in the pre-transpose local
                  // tensor is 1.
                  constexpr int64_t sourceReductionDim = 1;

                  if (inputTy.getRank() != 2)
                    return Value();

                  SmallVector<mlir::utils::IteratorType> iteratorTypes;
                  iteratorTypes.reserve(inputTy.getRank());
                  for (int64_t dim = 0; dim < inputTy.getRank(); ++dim) {
                    iteratorTypes.push_back(
                        dim == sourceReductionDim
                            ? mlir::utils::IteratorType::reduction
                            : mlir::utils::IteratorType::parallel);
                  }

                  MLIRContext *ctx = b.getContext();
                  SmallVector<AffineExpr> inputExprs;
                  SmallVector<AffineExpr> outputExprs;
                  inputExprs.reserve(inputTy.getRank());
                  outputExprs.reserve(expectedLocalTy.getRank());

                  for (int64_t dim = 0; dim < inputTy.getRank(); ++dim) {
                    auto expr = getAffineDimExpr(dim, ctx);
                    inputExprs.push_back(expr);
                    if (dim != sourceReductionDim)
                      outputExprs.push_back(expr);
                  }

                  AffineMap inputMap =
                      AffineMap::get(inputTy.getRank(), 0, inputExprs, ctx);
                  AffineMap outputMap =
                      AffineMap::get(inputTy.getRank(), 0, outputExprs, ctx);

                  auto localReduce = mlir::linalg::GenericOp::create(
                      b, reduce.getLoc(),
                      /*resultTensorTypes=*/TypeRange{expectedLocalTy},
                      /*inputs=*/ValueRange{localReduceInput},
                      /*outputs=*/ValueRange{localInit},
                      /*indexingMaps=*/
                      ArrayRef<AffineMap>{inputMap, outputMap},
                      /*iteratorTypes=*/iteratorTypes,
                      /*doc=*/StringRef(), /*libraryCall=*/StringRef());

                  if (!hasCompatibleRegionArgumentTypes(
                          reduce.getRegion().front(),
                          ValueRange{localReduceInput}, ValueRange{localInit}))
                    return Value();

                  cloneLinalgRegion(reduce.getRegion(),
                                    localReduce.getRegion());

                  Value localRes = localReduce.getResult(0);
                  cache[base] = localRes;
                  return localRes;
                }
              }
            }
          }

          RankedTensorType inputLocalTy =
              getReduceInputLocalType(reduce, expectedLocalTy);
          if (inputLocalTy) {
            Value localInput = materializeLocalValue(reduce.getInputs()[0],
                                                     inputLocalTy, cache);
            Value localInit = materializeLocalValue(reduce.getInits()[0],
                                                    expectedLocalTy, cache);
            if (localInput && localInit) {
              auto inputTy = dyn_cast<RankedTensorType>(localInput.getType());
              if (!inputTy)
                return Value();

              llvm::SmallSet<int64_t, 4> reducedDims;
              for (int64_t dim : reduce.getDimensions())
                (void)reducedDims.insert(dim);

              SmallVector<mlir::utils::IteratorType> iteratorTypes;
              iteratorTypes.reserve(inputTy.getRank());
              for (int64_t dim = 0; dim < inputTy.getRank(); ++dim) {
                iteratorTypes.push_back(
                    reducedDims.contains(dim)
                        ? mlir::utils::IteratorType::reduction
                        : mlir::utils::IteratorType::parallel);
              }

              MLIRContext *ctx = b.getContext();
              SmallVector<AffineExpr> inputExprs;
              SmallVector<AffineExpr> outputExprs;
              inputExprs.reserve(inputTy.getRank());
              outputExprs.reserve(expectedLocalTy.getRank());

              for (int64_t dim = 0; dim < inputTy.getRank(); ++dim) {
                auto expr = getAffineDimExpr(dim, ctx);
                inputExprs.push_back(expr);
                if (!reducedDims.contains(dim))
                  outputExprs.push_back(expr);
              }

              AffineMap inputMap =
                  AffineMap::get(inputTy.getRank(), 0, inputExprs, ctx);
              AffineMap outputMap =
                  AffineMap::get(inputTy.getRank(), 0, outputExprs, ctx);

              auto localReduce = mlir::linalg::GenericOp::create(
                  b, reduce.getLoc(),
                  /*resultTensorTypes=*/TypeRange{expectedLocalTy},
                  /*inputs=*/ValueRange{localInput},
                  /*outputs=*/ValueRange{localInit},
                  /*indexingMaps=*/ArrayRef<AffineMap>{inputMap, outputMap},
                  /*iteratorTypes=*/iteratorTypes,
                  /*doc=*/StringRef(), /*libraryCall=*/StringRef());

              if (!hasCompatibleRegionArgumentTypes(reduce.getRegion().front(),
                                                    ValueRange{localInput},
                                                    ValueRange{localInit}))
                return Value();

              cloneLinalgRegion(reduce.getRegion(), localReduce.getRegion());

              Value localRes = localReduce.getResult(0);
              cache[base] = localRes;
              return localRes;
            }
          }
        }

        // tensor.expand_shape: localize the source and rebuild the same
        // unit-expansion over the local tile.
        if (auto expand =
                dyn_cast_or_null<tensor::ExpandShapeOp>(base.getDefiningOp())) {
          RankedTensorType inputLocalTy =
              getExpandShapeInputLocalType(expand, expectedLocalTy);
          if (inputLocalTy) {
            Value localInput = materializeLocalValue(expand->getOperand(0),
                                                     inputLocalTy, cache);
            if (localInput) {
              SmallVector<ReassociationIndices> reassociation =
                  expand.getReassociationIndices();

              SmallVector<OpFoldResult> localOutputShape;
              localOutputShape.reserve(expectedLocalTy.getRank());
              for (int64_t dim : expectedLocalTy.getShape()) {
                if (ShapedType::isDynamic(dim) || dim <= 0)
                  return Value();
                localOutputShape.push_back(b.getIndexAttr(dim));
              }

              auto localExpand = tensor::ExpandShapeOp::create(
                  b, expand.getLoc(), expectedLocalTy, localInput,
                  reassociation, localOutputShape);

              Value localRes = localExpand->getResult(0);
              cache[base] = localRes;
              return localRes;
            }
          }
        }

        // linalg.generic: clone producer locally if compatible.
        if (auto prod = dyn_cast_or_null<mlir::linalg::GenericOp>(
                base.getDefiningOp())) {

          // Multi-result broadcast-aware producer.
          //
          // This covers patterns such as:
          //   %centered, %square = linalg.generic
          //     ins(%x, %mean : tensor<MxN>, tensor<M>)
          //     outs(%empty0, %empty1 : tensor<MxN>, tensor<MxN>)
          //
          // E.g., the final layer-norm generic may consume both
          // %centered and %square through different result numbers.
          // The multi-result producer once over localized operands,
          // then cache all local results:
          //   cache[%producer#0] = %local_producer#0
          //   cache[%producer#1] = %local_producer#1
          auto tryLocalizeMultiResultGeneric = [&]() -> Value {
            if (prod->getNumResults() <= 1)
              return Value();

            auto requestedResult = dyn_cast<OpResult>(base);
            if (!requestedResult ||
                requestedResult.getOwner() != prod.getOperation())
              return Value();

            const unsigned requestedResultNumber =
                requestedResult.getResultNumber();

            const int64_t numInputs = prod.getNumDpsInputs();
            const int64_t numOutputs = prod.getNumDpsInits();

            if (numInputs < 1 || numOutputs < 2)
              return Value();

            if (numOutputs != static_cast<int64_t>(prod->getNumResults()))
              return Value();

            if (requestedResultNumber >= static_cast<unsigned>(numOutputs))
              return Value();

            if (!expectedLocalTy ||
                !isSupportedIdentityElementwiseRank(expectedLocalTy.getRank()))
              return Value();

            if (prod.getNumLoops() != expectedLocalTy.getRank())
              return Value();

            auto iteratorTypes = prod.getIteratorTypesArray();
            if (static_cast<int64_t>(iteratorTypes.size()) !=
                expectedLocalTy.getRank())
              return Value();

            if (!llvm::all_of(iteratorTypes, [](mlir::utils::IteratorType it) {
                  return it == mlir::utils::IteratorType::parallel;
                }))
              return Value();

            auto maps = prod.getIndexingMapsArray();
            if (static_cast<int64_t>(maps.size()) != numInputs + numOutputs)
              return Value();

            // All tensor outputs must be identity maps for the requested rank.
            // This covers rank-2 producer chains and higher-rank elementwise
            // kernels that return multiple tensors.
            for (int64_t i = 0; i < numOutputs; ++i) {
              if (!maps[numInputs + i].isIdentity())
                return Value();
            }

            SmallVector<RankedTensorType> localResultTypes;
            localResultTypes.reserve(numOutputs);

            SmallVector<Type> resultTensorTypes;
            resultTensorTypes.reserve(numOutputs);

            for (Value result : prod->getResults()) {
              auto resultTy = dyn_cast<RankedTensorType>(result.getType());
              if (!resultTy || resultTy.getRank() != expectedLocalTy.getRank())
                return Value();

              RankedTensorType localResultTy =
                  getLocalTypeWithShape(resultTy, expectedLocalTy.getShape());
              if (!localResultTy)
                return Value();

              localResultTypes.push_back(localResultTy);
              resultTensorTypes.push_back(localResultTy);
            }

            // The caller requested a specific result. We Make sure the
            // requested local type is exactly the one the caller expects.
            if (localResultTypes[requestedResultNumber] != expectedLocalTy)
              return Value();

            SmallVector<Value> localInputs;
            localInputs.reserve(numInputs);

            for (int64_t i = 0; i < numInputs; ++i) {
              OpOperand *in = prod.getDpsInputOperand(i);
              auto inTy = dyn_cast<RankedTensorType>(in->get().getType());
              if (!inTy)
                return Value();

              AffineMap inputMap = maps[i];
              RankedTensorType inputLocalTy;

              // Identity inputs shard like the results, preserving the input
              // element type.
              if (inTy.getRank() == expectedLocalTy.getRank() &&
                  inputMap.isIdentity()) {
                inputLocalTy =
                    getLocalTypeWithShape(inTy, expectedLocalTy.getShape());
              }

              // Rank-1 broadcast inputs use the same rule as the final
              // broadcast-aware rank-2 consumer:
              //   row broadcast    (d0, d1) -> (d0): tensor<M> -> tensor<tileM>
              //   column broadcast (d0, d1) -> (d1): tensor<N> -> tensor<N>
              if (!inputLocalTy && expectedLocalTy.getRank() == 2 &&
                  inTy.getRank() == 1 &&
                  (isRowBroadcastMap(inputMap) ||
                   isColumnBroadcastMap(inputMap))) {
                inputLocalTy = getExpectedLocalTypeForOperand(inTy, inputMap,
                                                              expectedLocalTy);
              }

              if (!inputLocalTy)
                return Value();

              Value localInput =
                  materializeLocalValue(in->get(), inputLocalTy, cache);
              if (!localInput)
                return Value();

              localInputs.push_back(localInput);
            }

            SmallVector<Value> localOutputs;
            localOutputs.reserve(numOutputs);
            for (RankedTensorType localResultTy : localResultTypes)
              localOutputs.push_back(
                  buildTensorEmptyLike(prod.getLoc(), localResultTy));

            auto localProd = mlir::linalg::GenericOp::create(
                b, prod.getLoc(),
                /*resultTensorTypes=*/TypeRange(resultTensorTypes),
                /*inputs=*/ValueRange(localInputs),
                /*outputs=*/ValueRange(localOutputs),
                /*indexingMaps=*/prod.getIndexingMapsArray(),
                /*iteratorTypes=*/prod.getIteratorTypesArray(),
                /*doc=*/StringRef(), /*libraryCall=*/StringRef());

            if (!hasCompatibleRegionArgumentTypes(prod.getRegion().front(),
                                                  localInputs, localOutputs))
              return Value();

            cloneLinalgRegion(prod.getRegion(), localProd.getRegion());

            for (auto [oldResult, newResult] :
                 llvm::zip(prod->getResults(), localProd->getResults()))
              cache[oldResult] = newResult;

            return localProd->getResult(requestedResultNumber);
          };

          if (Value localMultiResult = tryLocalizeMultiResultGeneric())
            return localMultiResult;

          if (isShapePreservingElementwiseGeneric(prod, expectedLocalTy)) {
            SmallVector<Value> prodInputs;
            const int64_t nIn = prod.getNumDpsInputs();
            prodInputs.reserve(nIn);
            for (int64_t i = 0; i < nIn; ++i) {
              Value inV = prod.getDpsInputOperand(i)->get();
              auto inTy = dyn_cast<RankedTensorType>(inV.getType());
              if (!inTy)
                return Value();

              // Preserve the input element type.  Some producer chains
              // contain type-changing generics, for example f16 -> f32.
              // The local result type is driven by expectedLocalTy, but each
              // localized input must keep the element type of the original
              // input tensor.  Reusing expectedLocalTy for inputs would
              // incorrectly turn tensor<...xf16> inputs into tensor<...xf32>
              // and may make the cloned region yield a value whose type no
              // longer matches the enclosing linalg.generic output.
              RankedTensorType inputLocalTy =
                  getLocalTypeWithShape(inTy, expectedLocalTy.getShape());
              if (!inputLocalTy)
                return Value();

              prodInputs.push_back(
                  materializeLocalValue(inV, inputLocalTy, cache));
            }

            Value outLocalInit = mlir::tensor::EmptyOp::create(
                b, prod.getLoc(), expectedLocalTy.getShape(),
                expectedLocalTy.getElementType());

            auto newProd = mlir::linalg::GenericOp::create(
                b, prod.getLoc(),
                /*resultTensorTypes=*/TypeRange{expectedLocalTy},
                /*inputs=*/ValueRange{prodInputs},
                /*outputs=*/ValueRange{outLocalInit},
                /*indexingMaps=*/prod.getIndexingMapsArray(),
                /*iteratorTypes=*/prod.getIteratorTypesArray(),
                /*doc=*/StringRef(), /*libraryCall=*/StringRef());

            if (!hasCompatibleRegionArgumentTypes(prod.getRegion().front(),
                                                  prodInputs,
                                                  ValueRange{outLocalInit}))
              return Value();

            cloneLinalgRegion(prod.getRegion(), newProd.getRegion());

            Value localRes = newProd.getResult(0);
            cache[base] = localRes;
            return localRes;
          }

          // Broadcast-aware rank-2 elementwise producer:
          //   tensor<MxN>, tensor<M>, tensor<N> -> tensor<MxN>
          //
          // When expectedLocalTy is tensor<localM x N>, rank-2 identity inputs
          // are localized to tensor<localM x N>, row-broadcast inputs are
          // localized to tensor<localM>, and column-broadcast inputs remain
          // tensor<N> replicated on every participant. This covers the final
          // blocked-layer-norm generic:
          //   x[row, col], rstd[row], gamma[col], beta[col]
          auto prodResTy =
              dyn_cast<RankedTensorType>(prod.getResult(0).getType());
          auto prodMaps = prod.getIndexingMapsArray();
          if (isSupportedBroadcast2DGeneric(
                  prod, prodResTy,
                  [&]() {
                    SmallVector<RankedTensorType> tys;
                    tys.reserve(prod.getNumDpsInputs());
                    for (OpOperand *in : prod.getDpsInputOperands())
                      tys.push_back(
                          dyn_cast<RankedTensorType>(in->get().getType()));
                    return tys;
                  }()) &&
              expectedLocalTy.getRank() == 2) {
            SmallVector<Value> prodInputs;
            prodInputs.reserve(prod.getNumDpsInputs());

            for (OpOperand *in : prod.getDpsInputOperands()) {
              auto inTy = dyn_cast<RankedTensorType>(in->get().getType());
              if (!inTy)
                return Value();

              RankedTensorType inputLocalTy = getExpectedLocalTypeForOperand(
                  inTy, prodMaps[in->getOperandNumber()], expectedLocalTy);
              if (!inputLocalTy)
                return Value();

              prodInputs.push_back(
                  materializeLocalValue(in->get(), inputLocalTy, cache));
            }

            Value outLocalInit =
                buildTensorEmptyLike(prod.getLoc(), expectedLocalTy);
            auto newProd = mlir::linalg::GenericOp::create(
                b, prod.getLoc(),
                /*resultTensorTypes=*/TypeRange{expectedLocalTy},
                /*inputs=*/ValueRange{prodInputs},
                /*outputs=*/ValueRange{outLocalInit},
                /*indexingMaps=*/prod.getIndexingMapsArray(),
                /*iteratorTypes=*/prod.getIteratorTypesArray(),
                /*doc=*/StringRef(), /*libraryCall=*/StringRef());

            if (!hasCompatibleRegionArgumentTypes(prod.getRegion().front(),
                                                  prodInputs,
                                                  ValueRange{outLocalInit}))
              return Value();

            cloneLinalgRegion(prod.getRegion(), newProd.getRegion());

            Value localRes = newProd.getResult(0);
            cache[base] = localRes;
            return localRes;
          }
        }

        // scf.for with tensor iter_args: build a shard-local clone of the
        // loop on demand and return the corresponding local loop result.
        //
        // This path does not distribute the IV. It keeps the recurrence loop
        // intact on every NSP thread, but carries only shard-local tensor
        // state through the loop. This is the non-collective strategy for
        // online recurrences over K/V blocks.
        if (auto forOp =
                dyn_cast_or_null<mlir::scf::ForOp>(base.getDefiningOp())) {
          if (succeeded(localizeScfForWithTensorIterArgs(forOp, cache))) {
            auto localIt = cache.find(base);
            if (localIt != cache.end() &&
                localIt->second.getType() == expectedLocalTy)
              return localIt->second;
          }
        }

        // Fallback: slice the global value.
        Value local = mlir::shard::AllSliceOp::create(
                          b, base.getLoc(), /*result_type=*/expectedLocalTy,
                          /*input=*/base,
                          /*grid=*/"nsp",
                          /*gridAxes=*/gridAxes,
                          /*sliceAxis=*/splitAxis)
                          .getResult();
        cache[base] = local;
        return local;
      };

      // Build a shard-local clone of an scf.for with tensor iter_args.
      //
      // The helper is called from materializeLocalValue when a downstream
      // localized consumer requests a local version of one of the loop
      // results. This avoids replacing the original loop results with
      // different-typed local values.
      localizeScfForWithTensorIterArgs =
          [&](mlir::scf::ForOp forOp,
              llvm::DenseMap<Value, Value> &cache) -> LogicalResult {
        if (!forOp || forOp.getResults().empty() || forOp.getInitArgs().empty())
          return failure();

        if (isNspDistributedLoop(forOp) ||
            hasNspDistributedLoopAncestor(forOp.getOperation()))
          return failure();

        if (auto cached = localizedScfForCache.lookup(forOp.getOperation())) {
          for (auto [oldResult, newResult] :
               llvm::zip(forOp.getResults(), cached.getResults()))
            cache[oldResult] = newResult;
          return success();
        }

        if (failedLocalizedScfForCache.contains(forOp.getOperation()))
          return failure();

        constexpr int64_t loopSplitAxis = 0;

        auto getAxisLocalType = [&](RankedTensorType globalTy,
                                    int64_t axis) -> RankedTensorType {
          if (!globalTy || globalTy.getRank() == 0 || axis < 0 ||
              axis >= globalTy.getRank())
            return RankedTensorType();
          int64_t dim = globalTy.getDimSize(axis);
          if (ShapedType::isDynamic(dim) || dim <= 0 || dim % numShards != 0)
            return RankedTensorType();
          SmallVector<int64_t> shape(globalTy.getShape().begin(),
                                     globalTy.getShape().end());
          shape[axis] = dim / numShards;
          return RankedTensorType::get(shape, globalTy.getElementType(),
                                       globalTy.getEncoding());
        };

        SmallVector<char> localizeResult(forOp.getNumResults(), false);
        SmallVector<RankedTensorType> localResultTypes(forOp.getNumResults());
        llvm::SmallSet<int64_t, 4> shardedLeadingExtents;

        for (auto [idx, it] : llvm::enumerate(
                 llvm::zip(forOp.getInitArgs(), forOp.getResults()))) {
          Value init = std::get<0>(it);
          Value result = std::get<1>(it);

          auto initTy = dyn_cast<RankedTensorType>(init.getType());
          auto resultTy = dyn_cast<RankedTensorType>(result.getType());
          if (!initTy || !resultTy || initTy != resultTy)
            continue;

          RankedTensorType localTy = getAxisLocalType(resultTy, loopSplitAxis);
          if (!localTy)
            continue;

          localizeResult[idx] = true;
          localResultTypes[idx] = localTy;
          (void)shardedLeadingExtents.insert(
              resultTy.getDimSize(loopSplitAxis));
        }

        if (!llvm::any_of(localizeResult, [](char v) { return v; }))
          return failure();

        auto isLoopRowShardedType = [&](RankedTensorType ty) -> bool {
          if (!ty || ty.getRank() == 0)
            return false;
          int64_t dim0 = ty.getDimSize(loopSplitAxis);
          return !ShapedType::isDynamic(dim0) &&
                 shardedLeadingExtents.contains(dim0);
        };

        auto getLoopRowLocalType =
            [&](RankedTensorType ty) -> RankedTensorType {
          if (!isLoopRowShardedType(ty))
            return RankedTensorType();
          return getAxisLocalType(ty, loopSplitAxis);
        };

        auto isDefinedInsideLoop = [&](Value v) -> bool {
          Operation *def = v.getDefiningOp();
          return def && def->getParentOfType<mlir::scf::ForOp>() == forOp;
        };

        OpBuilder preBuilder(forOp);
        Location loc = forOp.getLoc();

        SmallVector<Operation *> speculativeOps;
        llvm::DenseMap<Value, Value> externalLocalCache;
        auto getOrCreateExternalLocal = [&](Value v,
                                            RankedTensorType localTy) -> Value {
          if (!v || !localTy)
            return Value();
          if (v.getType() == localTy)
            return v;
          if (isDefinedInsideLoop(v))
            return Value();
          if (Value cached = externalLocalCache.lookup(v))
            return cached;

          auto globalTy = dyn_cast<RankedTensorType>(v.getType());
          if (!globalTy)
            return Value();

          auto allSlice = mlir::shard::AllSliceOp::create(
              preBuilder, v.getLoc(), /*result_type=*/localTy,
              /*input=*/v,
              /*grid=*/"nsp",
              /*gridAxes=*/gridAxes,
              /*sliceAxis=*/loopSplitAxis);
          Value local = allSlice.getResult();
          speculativeOps.push_back(allSlice.getOperation());
          externalLocalCache[v] = local;
          return local;
        };

        SmallVector<Value> newInitArgs;
        newInitArgs.reserve(forOp.getInitArgs().size());
        for (auto [idx, init] : llvm::enumerate(forOp.getInitArgs())) {
          if (!localizeResult[idx]) {
            newInitArgs.push_back(init);
            continue;
          }

          Value localInit =
              getOrCreateExternalLocal(init, localResultTypes[idx]);
          if (!localInit)
            return failure();
          newInitArgs.push_back(localInit);
        }

        auto newFor = mlir::scf::ForOp::create(
            preBuilder, loc, forOp.getLowerBound(), forOp.getUpperBound(),
            forOp.getStep(), newInitArgs);
        newFor->setAttr("nsp.localized_loop", preBuilder.getUnitAttr());
        localizedScfForCache[forOp.getOperation()] = newFor;

        auto rollbackLocalizedLoop = [&]() {
          newFor.erase();
          localizedScfForCache.erase(forOp.getOperation());
          failedLocalizedScfForCache.insert(forOp.getOperation());

          // External all_slice values are speculative until the localized loop
          // body has been fully cloned. If cloning fails, erase them as well so
          // the pass does not leave dead slice anchors for canonicalization to
          // clean up later.
          for (Operation *op : llvm::reverse(speculativeOps)) {
            if (op && op->use_empty())
              op->erase();
          }
        };

        Block *oldBody = forOp.getBody();
        Block *newBody = newFor.getBody();
        newBody->getOperations().clear();

        IRMapping mapping;
        mapping.map(forOp.getInductionVar(), newFor.getInductionVar());
        for (auto [oldArg, newArg] :
             llvm::zip(forOp.getRegionIterArgs(), newFor.getRegionIterArgs()))
          mapping.map(oldArg, newArg);

        auto lookupMappedOrSelf = [&](Value v) -> Value {
          if (Value mapped = mapping.lookupOrNull(v))
            return mapped;
          return v;
        };

        auto cloneRegionInto = [&](Region &src, Region &dst) {
          dst.getBlocks().clear();

          Block &srcBlock = src.front();
          Block *dstBlock = new Block();
          dst.push_back(dstBlock);

          for (BlockArgument arg : srcBlock.getArguments())
            dstBlock->addArgument(arg.getType(), arg.getLoc());

          IRMapping regionMap;
          for (auto [srcArg, dstArg] :
               llvm::zip(srcBlock.getArguments(), dstBlock->getArguments()))
            regionMap.map(srcArg, dstArg);

          OpBuilder nb = OpBuilder::atBlockEnd(dstBlock);
          for (Operation &nested : srcBlock.getOperations())
            nb.clone(nested, regionMap);
        };

        auto remapOperandForMap = [&](Value operand, AffineMap map,
                                      int64_t shardedLoopDim) -> Value {
          Value mapped = lookupMappedOrSelf(operand);

          auto operandTy = dyn_cast<RankedTensorType>(operand.getType());
          if (!operandTy || shardedLoopDim < 0)
            return mapped;

          for (auto [operandDim, expr] : llvm::enumerate(map.getResults())) {
            auto dimExpr = dyn_cast<AffineDimExpr>(expr);
            if (!dimExpr || dimExpr.getPosition() != shardedLoopDim)
              continue;

            // If this operand was already produced by an earlier localized op
            // in the cloned loop, keep the mapped value. This covers cases
            // where a transpose or other structured op moves the sharded loop
            // dimension away from physical tensor dimension 0.
            if (static_cast<int64_t>(operandDim) != loopSplitAxis) {
              if (mapped.getType() != operand.getType())
                return mapped;
              return Value();
            }

            RankedTensorType localTy = getLoopRowLocalType(operandTy);
            if (!localTy)
              return Value();

            if (mapped.getType() == localTy)
              return mapped;
            return getOrCreateExternalLocal(operand, localTy);
          }

          return mapped;
        };

        auto cloneGeneric = [&](mlir::linalg::GenericOp generic,
                                OpBuilder &bodyBuilder) -> LogicalResult {
          const int64_t numInputs = generic.getNumDpsInputs();
          const int64_t numOutputs = generic.getNumDpsInits();
          auto maps = generic.getIndexingMapsArray();

          SmallVector<Value> newOutputs;
          newOutputs.reserve(numOutputs);
          bool hasLocalizedOutput = false;
          int64_t shardedLoopDim = -1;

          for (int64_t outIdx = 0; outIdx < numOutputs; ++outIdx) {
            Value oldInit = generic.getDpsInitOperand(outIdx)->get();
            Value newInit = lookupMappedOrSelf(oldInit);

            auto oldResultTy =
                dyn_cast<RankedTensorType>(generic.getResult(outIdx).getType());
            if (oldResultTy) {
              RankedTensorType localTy = getLoopRowLocalType(oldResultTy);
              if (localTy && newInit.getType() != localTy) {
                newInit = getOrCreateExternalLocal(oldInit, localTy);
                if (!newInit)
                  return failure();
              }
            }

            newOutputs.push_back(newInit);

            auto newInitTy = dyn_cast<RankedTensorType>(newInit.getType());
            if (!oldResultTy || !newInitTy || oldResultTy == newInitTy)
              continue;

            hasLocalizedOutput = true;

            AffineMap outMap = maps[numInputs + outIdx];
            if (!outMap || outMap.getNumResults() <= loopSplitAxis)
              return failure();
            auto d0 = dyn_cast<AffineDimExpr>(outMap.getResult(loopSplitAxis));
            if (!d0)
              return failure();

            if (shardedLoopDim < 0)
              shardedLoopDim = d0.getPosition();
            else if (shardedLoopDim != d0.getPosition())
              return failure();
          }

          if (!hasLocalizedOutput) {
            Operation *cloned = bodyBuilder.clone(*generic, mapping);
            for (auto [oldResult, newResult] :
                 llvm::zip(generic->getResults(), cloned->getResults()))
              mapping.map(oldResult, newResult);
            return success();
          }

          SmallVector<Value> newInputs;
          newInputs.reserve(numInputs);
          for (int64_t inIdx = 0; inIdx < numInputs; ++inIdx) {
            Value oldInput = generic.getDpsInputOperand(inIdx)->get();
            Value newInput =
                remapOperandForMap(oldInput, maps[inIdx], shardedLoopDim);
            if (!newInput)
              return failure();
            newInputs.push_back(newInput);
          }

          SmallVector<Type> resultTypes;
          resultTypes.reserve(newOutputs.size());
          for (Value out : newOutputs)
            resultTypes.push_back(out.getType());

          auto newGeneric = mlir::linalg::GenericOp::create(
              bodyBuilder, generic.getLoc(),
              /*resultTensorTypes=*/TypeRange(resultTypes),
              /*inputs=*/ValueRange(newInputs),
              /*outputs=*/ValueRange(newOutputs),
              /*indexingMaps=*/generic.getIndexingMapsArray(),
              /*iteratorTypes=*/generic.getIteratorTypesArray(),
              /*doc=*/StringRef(), /*libraryCall=*/StringRef());

          cloneRegionInto(generic.getRegion(), newGeneric.getRegion());

          for (auto [oldResult, newResult] :
               llvm::zip(generic->getResults(), newGeneric->getResults()))
            mapping.map(oldResult, newResult);
          return success();
        };

        auto cloneReduce = [&](mlir::linalg::ReduceOp reduce,
                               OpBuilder &bodyBuilder) -> LogicalResult {
          if (reduce.getInputs().size() != 1 || reduce.getInits().size() != 1)
            return failure();

          Value newInput = lookupMappedOrSelf(reduce.getInputs()[0]);
          Value oldInit = reduce.getInits()[0];
          Value newInit = lookupMappedOrSelf(oldInit);

          auto oldResultTy =
              dyn_cast<RankedTensorType>(reduce->getResult(0).getType());
          if (oldResultTy) {
            RankedTensorType localTy = getLoopRowLocalType(oldResultTy);
            if (localTy && newInit.getType() != localTy) {
              newInit = getOrCreateExternalLocal(oldInit, localTy);
              if (!newInit)
                return failure();
            }
          }

          auto newInitTy = dyn_cast<RankedTensorType>(newInit.getType());
          if (!oldResultTy || !newInitTy || oldResultTy == newInitTy) {
            Operation *cloned = bodyBuilder.clone(*reduce, mapping);
            mapping.map(reduce->getResult(0), cloned->getResult(0));
            return success();
          }

          auto inputTy = dyn_cast<RankedTensorType>(newInput.getType());
          if (!inputTy)
            return failure();

          llvm::SmallSet<int64_t, 4> reducedDims;
          for (int64_t dim : reduce.getDimensions())
            (void)reducedDims.insert(dim);

          SmallVector<mlir::utils::IteratorType> iteratorTypes;
          iteratorTypes.reserve(inputTy.getRank());
          for (int64_t dim = 0; dim < inputTy.getRank(); ++dim) {
            iteratorTypes.push_back(reducedDims.contains(dim)
                                        ? mlir::utils::IteratorType::reduction
                                        : mlir::utils::IteratorType::parallel);
          }

          MLIRContext *ctx = bodyBuilder.getContext();
          SmallVector<AffineExpr> inputExprs;
          SmallVector<AffineExpr> outputExprs;
          inputExprs.reserve(inputTy.getRank());
          outputExprs.reserve(newInitTy.getRank());

          for (int64_t dim = 0; dim < inputTy.getRank(); ++dim) {
            auto expr = getAffineDimExpr(dim, ctx);
            inputExprs.push_back(expr);
            if (!reducedDims.contains(dim))
              outputExprs.push_back(expr);
          }

          AffineMap inputMap =
              AffineMap::get(inputTy.getRank(), 0, inputExprs, ctx);
          AffineMap outputMap =
              AffineMap::get(inputTy.getRank(), 0, outputExprs, ctx);

          auto newReduce = mlir::linalg::GenericOp::create(
              bodyBuilder, reduce.getLoc(),
              /*resultTensorTypes=*/TypeRange{newInitTy},
              /*inputs=*/ValueRange{newInput},
              /*outputs=*/ValueRange{newInit},
              /*indexingMaps=*/ArrayRef<AffineMap>{inputMap, outputMap},
              /*iteratorTypes=*/iteratorTypes,
              /*doc=*/StringRef(), /*libraryCall=*/StringRef());

          cloneRegionInto(reduce.getRegion(), newReduce.getRegion());
          mapping.map(reduce->getResult(0), newReduce.getResult(0));
          return success();
        };

        auto cloneSingleResultNamedLinalg =
            [&](mlir::linalg::LinalgOp linalgOp,
                OpBuilder &bodyBuilder) -> LogicalResult {
          Operation *oldOp = linalgOp.getOperation();
          if (isa<mlir::linalg::GenericOp, mlir::linalg::ReduceOp,
                  mlir::linalg::FillOp, mlir::linalg::TransposeOp>(oldOp))
            return failure();

          if (linalgOp.getNumDpsInits() != 1 || oldOp->getNumResults() != 1)
            return failure();

          auto maps = linalgOp.getIndexingMapsArray();
          if (static_cast<int64_t>(maps.size()) !=
              linalgOp.getNumDpsInputs() + linalgOp.getNumDpsInits())
            return failure();

          Value oldOutput = linalgOp.getDpsInitOperand(0)->get();
          Value newOutput = lookupMappedOrSelf(oldOutput);

          auto oldResultTy =
              dyn_cast<RankedTensorType>(oldOp->getResult(0).getType());
          if (oldResultTy) {
            RankedTensorType localTy = getLoopRowLocalType(oldResultTy);
            if (localTy && newOutput.getType() != localTy) {
              newOutput = getOrCreateExternalLocal(oldOutput, localTy);
              if (!newOutput)
                return failure();
            }
          }

          auto newOutputTy = dyn_cast<RankedTensorType>(newOutput.getType());
          if (!oldResultTy || !newOutputTy || oldResultTy == newOutputTy) {
            Operation *cloned = bodyBuilder.clone(*oldOp, mapping);
            mapping.map(oldOp->getResult(0), cloned->getResult(0));
            return success();
          }

          AffineMap outMap = maps[linalgOp.getNumDpsInputs()];
          if (!outMap || outMap.getNumResults() <= loopSplitAxis)
            return failure();

          auto d0 = dyn_cast<AffineDimExpr>(outMap.getResult(loopSplitAxis));
          if (!d0)
            return failure();
          int64_t shardedLoopDim = d0.getPosition();

          SmallVector<Value> newInputs;
          newInputs.reserve(linalgOp.getNumDpsInputs());
          for (int64_t inIdx = 0; inIdx < linalgOp.getNumDpsInputs(); ++inIdx) {
            Value oldInput = linalgOp.getDpsInputOperand(inIdx)->get();
            Value newInput =
                remapOperandForMap(oldInput, maps[inIdx], shardedLoopDim);
            if (!newInput)
              return failure();
            newInputs.push_back(newInput);
          }

          OperationState state(oldOp->getLoc(), oldOp->getName());
          state.addOperands(newInputs);
          state.addOperands(ValueRange{newOutput});
          state.addTypes(newOutput.getType());
          state.addAttributes(oldOp->getAttrs());

          Operation *newOp = bodyBuilder.create(state);
          mapping.map(oldOp->getResult(0), newOp->getResult(0));
          return success();
        };

        auto cloneMatmul = [&](mlir::linalg::MatmulOp matmul,
                               OpBuilder &bodyBuilder) -> LogicalResult {
          mlir::linalg::LinalgOp linalgOp(matmul.getOperation());
          if (linalgOp.getNumDpsInputs() != 2 ||
              linalgOp.getNumDpsInits() != 1 || matmul->getNumResults() != 1)
            return failure();

          auto maps = linalgOp.getIndexingMapsArray();
          if (maps.size() != 3)
            return failure();

          Value oldOutput = linalgOp.getDpsInitOperand(0)->get();
          Value newOutput = lookupMappedOrSelf(oldOutput);

          auto oldResultTy =
              dyn_cast<RankedTensorType>(matmul->getResult(0).getType());
          if (oldResultTy) {
            RankedTensorType localTy = getLoopRowLocalType(oldResultTy);
            if (localTy && newOutput.getType() != localTy) {
              newOutput = getOrCreateExternalLocal(oldOutput, localTy);
              if (!newOutput)
                return failure();
            }
          }

          auto newOutputTy = dyn_cast<RankedTensorType>(newOutput.getType());
          if (!oldResultTy || !newOutputTy || oldResultTy == newOutputTy) {
            Operation *cloned = bodyBuilder.clone(*matmul, mapping);
            mapping.map(matmul->getResult(0), cloned->getResult(0));
            return success();
          }

          AffineMap outMap = maps.back();
          if (!outMap || outMap.getNumResults() <= loopSplitAxis)
            return failure();

          auto d0 = dyn_cast<AffineDimExpr>(outMap.getResult(loopSplitAxis));
          if (!d0)
            return failure();
          int64_t shardedLoopDim = d0.getPosition();

          SmallVector<Value> newInputs;
          newInputs.reserve(linalgOp.getNumDpsInputs());
          for (int64_t inIdx = 0; inIdx < linalgOp.getNumDpsInputs(); ++inIdx) {
            Value oldInput = linalgOp.getDpsInputOperand(inIdx)->get();
            Value newInput =
                remapOperandForMap(oldInput, maps[inIdx], shardedLoopDim);
            if (!newInput)
              return failure();
            newInputs.push_back(newInput);
          }

          // Named linalg ops such as linalg.matmul do not have a generic
          // region to clone. Rebuild the same operation name with remapped
          // DPS operands and local tensor result type. Copying the original
          // attributes preserves segment sizes and any named-op attributes.
          OperationState state(matmul.getLoc(), matmul->getName());
          state.addOperands(newInputs);
          state.addOperands(ValueRange{newOutput});
          state.addTypes(newOutput.getType());
          state.addAttributes(matmul->getAttrs());

          Operation *newOp = bodyBuilder.create(state);
          mapping.map(matmul->getResult(0), newOp->getResult(0));
          return success();
        };

        auto cloneOp = [&](Operation &op,
                           OpBuilder &bodyBuilder) -> LogicalResult {
          if (auto empty = dyn_cast<tensor::EmptyOp>(op)) {
            auto oldTy = dyn_cast<RankedTensorType>(empty.getType());
            RankedTensorType localTy = getLoopRowLocalType(oldTy);
            if (localTy) {
              Value localEmpty = tensor::EmptyOp::create(
                  bodyBuilder, empty.getLoc(), localTy.getShape(),
                  localTy.getElementType());
              mapping.map(empty.getResult(), localEmpty);
              return success();
            }
          }

          if (auto fill = dyn_cast<mlir::linalg::FillOp>(op)) {
            if (fill.getOutputs().size() != 1)
              return failure();
            Value oldOutput = fill.getOutputs()[0];
            Value newOutput = lookupMappedOrSelf(oldOutput);
            if (auto resultTy =
                    dyn_cast<RankedTensorType>(fill->getResult(0).getType())) {
              RankedTensorType localTy = getLoopRowLocalType(resultTy);
              if (localTy && newOutput.getType() != localTy) {
                newOutput = getOrCreateExternalLocal(oldOutput, localTy);
                if (!newOutput)
                  return failure();
              }
            }
            if (newOutput.getType() != oldOutput.getType()) {
              auto newFill = mlir::linalg::FillOp::create(
                  bodyBuilder, fill.getLoc(),
                  /*inputs=*/
                  ValueRange{lookupMappedOrSelf(fill.getInputs()[0])},
                  /*outputs=*/ValueRange{newOutput});
              mapping.map(fill->getResult(0), newFill->getResult(0));
              return success();
            }
          }

          if (auto transpose = dyn_cast<mlir::linalg::TransposeOp>(op)) {
            Value oldInput = transpose.getInput();
            Value oldInit = transpose.getInit();
            Value newInput = lookupMappedOrSelf(oldInput);
            Value newInit = lookupMappedOrSelf(oldInit);

            auto oldInputTy = dyn_cast<RankedTensorType>(oldInput.getType());
            auto newInputTy = dyn_cast<RankedTensorType>(newInput.getType());
            auto oldResultTy = dyn_cast<RankedTensorType>(
                transpose->getResult(0).getType());

            // If the input was localized by an earlier op in the cloned loop,
            // rebuild the transpose result type by applying the original
            // permutation to the localized input shape.
            if (oldInputTy && newInputTy && oldResultTy &&
                newInputTy != oldInputTy) {
              auto permutation = transpose.getPermutation();
              if ((int64_t)permutation.size() != newInputTy.getRank())
                return failure();

              SmallVector<int64_t> localResultShape;
              localResultShape.reserve(newInputTy.getRank());
              for (int64_t inputDim : permutation) {
                if (inputDim < 0 || inputDim >= newInputTy.getRank())
                  return failure();
                int64_t extent = newInputTy.getDimSize(inputDim);
                if (ShapedType::isDynamic(extent) || extent <= 0)
                  return failure();
                localResultShape.push_back(extent);
              }

              RankedTensorType localResultTy = RankedTensorType::get(
                  localResultShape, oldResultTy.getElementType(),
                  oldResultTy.getEncoding());

              if (newInit.getType() != localResultTy) {
                newInit = tensor::EmptyOp::create(
                    bodyBuilder, transpose.getLoc(), localResultTy.getShape(),
                    localResultTy.getElementType());
              }

              auto newTranspose = mlir::linalg::TransposeOp::create(
                  bodyBuilder, transpose.getLoc(),
                  /*input=*/newInput,
                  /*init=*/newInit,
                  /*permutation=*/permutation);
              mapping.map(transpose->getResult(0), newTranspose.getResult()[0]);
              return success();
            }

            // Replicated transposes are still valid in a localized loop.
            Operation *cloned = bodyBuilder.clone(op, mapping);
            mapping.map(transpose->getResult(0), cloned->getResult(0));
            return success();
          }

          if (auto matmul = dyn_cast<mlir::linalg::MatmulOp>(op))
            return cloneMatmul(matmul, bodyBuilder);

          if (auto generic = dyn_cast<mlir::linalg::GenericOp>(op))
            return cloneGeneric(generic, bodyBuilder);

          if (auto reduce = dyn_cast<mlir::linalg::ReduceOp>(op))
            return cloneReduce(reduce, bodyBuilder);

          if (auto linalgOp = dyn_cast<mlir::linalg::LinalgOp>(op))
            return cloneSingleResultNamedLinalg(linalgOp, bodyBuilder);

          bool hasRowShardedTensorResult = false;
          for (Type t : op.getResultTypes()) {
            if (getLoopRowLocalType(dyn_cast<RankedTensorType>(t))) {
              hasRowShardedTensorResult = true;
              break;
            }
          }
          if (hasRowShardedTensorResult)
            return failure();

          Operation *cloned = bodyBuilder.clone(op, mapping);
          for (auto [oldResult, newResult] :
               llvm::zip(op.getResults(), cloned->getResults()))
            mapping.map(oldResult, newResult);
          return success();
        };

        OpBuilder bodyBuilder(newFor.getContext());
        for (Operation &op : oldBody->without_terminator()) {
          bodyBuilder.setInsertionPointToEnd(newBody);
          if (failed(cloneOp(op, bodyBuilder))) {
            op.emitRemark()
                << "NSPLocalize: failed to localize this op while cloning "
                   "scf.for with tensor iter_args: "
                << op.getName();
            newFor.emitRemark()
                << "NSPLocalize: failed to localize scf.for with tensor "
                   "iter_args; keeping original loop";
            rollbackLocalizedLoop();
            return failure();
          }
        }

        auto oldYield = cast<mlir::scf::YieldOp>(oldBody->getTerminator());
        SmallVector<Value> newYieldOperands;
        newYieldOperands.reserve(oldYield.getNumOperands());
        for (auto [idx, yielded] : llvm::enumerate(oldYield.getOperands())) {
          Value mappedYield = lookupMappedOrSelf(yielded);
          if (localizeResult[idx] &&
              mappedYield.getType() != localResultTypes[idx]) {
            rollbackLocalizedLoop();
            return failure();
          }
          newYieldOperands.push_back(mappedYield);
        }
        bodyBuilder.setInsertionPointToEnd(newBody);
        mlir::scf::YieldOp::create(bodyBuilder, oldYield.getLoc(),
                                   newYieldOperands);

        for (auto [oldResult, newResult] :
             llvm::zip(forOp.getResults(), newFor.getResults()))
          cache[oldResult] = newResult;

        return success();
      };

      // Use a worklist since we'll rewrite in-place.
      SmallVector<mlir::linalg::GenericOp> worklist;
      func.walk([&](mlir::linalg::GenericOp g) { worklist.push_back(g); });

      for (mlir::linalg::GenericOp g : worklist) {
        // If this generic is enclosed by a DOALL loop already distributed by
        // this pass, do not also apply tensor tile localization. The loop
        // schedule already partitions the destination writes, so emitting
        // nsp.materialize_tile here would double-partition the inner tensor.
        if (!allowCollectives &&
            hasNspDistributedLoopAncestor(g.getOperation()))
          continue;

        // Initial structural filter.
        const int64_t numInputs = g.getNumDpsInputs();
        if (numInputs < 1 || numInputs > 4 || g.getNumDpsInits() != 1)
          continue;

        SmallVector<Value> inputs;
        inputs.reserve(numInputs);
        for (int64_t i = 0; i < numInputs; ++i)
          inputs.push_back(g.getDpsInputOperand(i)->get());
        Value oldInit = g.getDpsInitOperand(0)->get();

        SmallVector<RankedTensorType> inputTys;
        inputTys.reserve(numInputs);
        for (Value v : inputs)
          inputTys.push_back(dyn_cast<RankedTensorType>(v.getType()));
        auto outResTy = dyn_cast<RankedTensorType>(g.getResult(0).getType());
        if (!outResTy)
          continue;
        // for (RankedTensorType t : inputTys)
        //   if (!t)
        //     continue;

        bool allRanked = true;
        for (RankedTensorType t : inputTys)
          allRanked &= static_cast<bool>(t);
        if (!allRanked)
          continue;

        bool supportsIdentityElementwise =
            isSupportedElementwiseGenericShape(g, outResTy, inputTys);
        bool supportsBroadcast2D =
            isSupportedBroadcast2DGeneric(g, outResTy, inputTys);

        if (!supportsIdentityElementwise && !supportsBroadcast2D)
          continue;

        std::optional<int64_t> splitAxisOr;
        if (supportsBroadcast2D) {
          // The rank-2 broadcast path models row/column projections relative to
          // axis 0. Keep that path conservative and preserve the existing
          // split-axis contract.
          if (isValidSplitAxis(outResTy, 0))
            splitAxisOr = 0;
        } else {
          splitAxisOr = chooseSplitAxis(outResTy);
        }

        if (!splitAxisOr) {
          g.emitError() << "NSPLocalize: cannot choose a split axis for result "
                        << outResTy << " with grid size " << numShards
                        << " (requires at least one positive static dimension "
                           "divisible by the grid size)";
          signalPassFailure();
          return;
        }
        splitAxis = *splitAxisOr;

        auto localTy = getLocalType(outResTy);
        if (!localTy) {
          g.emitError() << "NSPLocalize: cannot compute local tile type for "
                           "result "
                        << outResTy << " with grid size " << numShards
                        << " and split axis " << splitAxis;
          signalPassFailure();
          return;
        }

        // Inputs must shard to the same local shape as the result.
        // For broadcast-aware rank-2 generics, rank-1 inputs are considered
        // compatible when their shard-local extent matches the row dimension
        // of the output tile.
        bool compatibleLocalShapes = true;
        auto inputMaps = g.getIndexingMapsArray();
        for (auto [idx, t] : llvm::enumerate(inputTys)) {
          if (supportsBroadcast2D) {
            if (!isCompatibleInputForBroadcastLocalize(t, inputMaps[idx],
                                                       outResTy)) {
              compatibleLocalShapes = false;
              break;
            }
          } else {
            // Existing path: require the same shard-local shape.
            // Do not require full local type equality here: elementwise ops may
            // legitimately change element type (e.g. f16 -> f32) while still
            // being perfectly shard-compatible.
            if (!haveSameLocalShape(t, outResTy)) {
              compatibleLocalShapes = false;
              break;
            }
          }
        }
        if (!compatibleLocalShapes) {
          // Leave this op untouched and continue. Failing the whole pass here
          // keeps shard annotations alive and breaks later passes, while this
          // specific generic is simply not supported by the current bring-up
          // materialization.
          continue;
        }

        Location loc = g.getLoc();
        b.setInsertionPoint(g);

        // If this generic is already inside a loop explicitly distributed by
        // this pass, do not apply the non-collective "store-by-tile"
        // materialization path again.
        //
        // Rationale:
        //   In distributed DOALL patterns (e.g. softmax outer-loop
        //   distribution), the partitioning semantics are already carried by
        //   the surrounding scf.for schedule:
        //     lb'   = lb + procIdx * step
        //     step' = step * numShards
        if (!allowCollectives &&
            hasNspDistributedLoopAncestor(g.getOperation()))
          continue;

        // Non-collective path must ONLY rewrite ops that directly materialize
        // into a destination memref. Intermediate elementwise ops (common in
        // pipelines like softmax) must remain intact; otherwise, moving the
        // region body (takeBody) would leave the original op with an empty
        // region and trigger verifier errors.
        std::optional<std::pair<bufferization::MaterializeInDestinationOp,
                                SmallVector<Operation *>>>
            sink;
        if (!allowCollectives) {
          sink = findMaterializeSink(g.getResult(0));
          if (!sink) {
            // No materialization sink => this is an intermediate tensor.
            continue;
          }
        }

        // Slice inputs into per-core tiles.
        //   For chained elementwise patterns, attempt to clone compatible
        //   producers into local-tile compute instead of slicing full
        //   global intermediates.
        // Important:
        //   For broadcast-aware rank-2 generics, not every operand uses the
        //   same shard-local type as the result. In particular, rank-1 inputs
        //   with map (d0, d1) -> (d0) must localize to tensor<localRows>,
        //   not tensor<localRows x localCols>.
        llvm::DenseMap<Value, Value> localCache;
        SmallVector<Value> localInputs;
        localInputs.reserve(numInputs);
        auto maps = g.getIndexingMapsArray();
        for (int64_t i = 0; i < numInputs; ++i) {
          RankedTensorType expectedInputLocalTy;

          if (supportsBroadcast2D) {
            expectedInputLocalTy =
                getExpectedLocalTypeForOperand(inputTys[i], maps[i], localTy);
            if (!expectedInputLocalTy) {
              g.emitError()
                  << "NSPLocalize: cannot compute operand-local type for input "
                  << i << " of broadcast-aware generic";
              signalPassFailure();
              return;
            }
          } else {
            // Preserve the input element type while using the result-local
            // shape. This keeps type-changing elementwise generics valid.
            expectedInputLocalTy =
                getLocalTypeWithShape(inputTys[i], localTy.getShape());
            if (!expectedInputLocalTy) {
              g.emitError()
                  << "NSPLocalize: cannot compute operand-local type for input "
                  << i << " of identity elementwise generic";
              signalPassFailure();
              return;
            }
          }

          Value localIn = materializeLocalValue(inputs[i], expectedInputLocalTy,
                                                localCache);
          if (!localIn) {
            g.emitError() << "NSPLocalize: failed to materialize local input "
                          << i << " for tensor type " << inputTys[i];
            signalPassFailure();
            return;
          }
          localInputs.push_back(localIn);
        }

        // Localize this computation into shard-local tensor semantics.
        auto localizedOrFail = localizeGenericToTensor(
            b, g, localTy, localInputs, oldInit, splitAxis);
        if (failed(localizedOrFail)) {
          g.emitError() << "NSPLocalize: failed to build localized tensor "
                           "generic";
          signalPassFailure();
          return;
        }

        auto localizedGeneric = *localizedOrFail;
        Value localResult = localizedGeneric.getResult(0);

        auto rebuildPostSinkTensorChain =
            [&](Value baseLocal, ArrayRef<Operation *> wrappers) -> Value {
          Value curLocal = baseLocal;

          for (Operation *wrapper : wrappers) {
            if (isa<mlir::shard::ShardOp>(wrapper))
              continue;

            // tensor.cast was already treated as a removable wrapper in this
            // path. Keep the local value as-is; the later materialization
            // contract only needs the local ranked tensor value.
            if (isa<tensor::CastOp>(wrapper))
              continue;

            if (auto expand = dyn_cast<tensor::ExpandShapeOp>(wrapper)) {
              auto localInputTy =
                  dyn_cast<RankedTensorType>(curLocal.getType());
              if (!localInputTy)
                return Value();

              RankedTensorType localExpandTy =
                  getExpandShapeLocalResultType(expand, localInputTy);
              if (!localExpandTy)
                return Value();

              SmallVector<ReassociationIndices> reassociation =
                  expand.getReassociationIndices();

              SmallVector<OpFoldResult> localOutputShape;
              localOutputShape.reserve(localExpandTy.getRank());
              for (int64_t dim : localExpandTy.getShape()) {
                if (ShapedType::isDynamic(dim) || dim <= 0)
                  return Value();
                localOutputShape.push_back(b.getIndexAttr(dim));
              }

              auto localExpand = tensor::ExpandShapeOp::create(
                  b, expand.getLoc(), localExpandTy, curLocal, reassociation,
                  localOutputShape);
              curLocal = localExpand->getResult(0);
              continue;
            }

            wrapper->emitRemark()
                << "NSPLocalize: unsupported tensor wrapper in materialize "
                   "sink chain: "
                << wrapper->getName();
            return Value();
          }

          return curLocal;
        };

        if (allowCollectives) {
          const llvm::APInt splitAxisAP(/*numBits=*/64, /*val=*/splitAxis,
                                        /*isSigned=*/true);
          Value globalResult =
              mlir::shard::AllGatherOp::create(
                  b, loc,
                  /*result=*/outResTy,
                  /*grid=*/"nsp",
                  /*grid_axes=*/llvm::ArrayRef<int16_t>(gridAxesI16),
                  /*input=*/localResult,
                  /*gather_axis=*/splitAxisAP)
                  .getResult();

          g.getResult(0).replaceAllUsesWith(globalResult);

          localizedGeneric->removeAttr("nsp.localized");
          localizedGeneric->removeAttr("nsp.materialize_tile_shape");
          localizedGeneric->removeAttr("nsp.materialize_split_axis");

          g.erase();
          continue;
        }

        // In non-collective mode, replace the old split-pass contract
        // ("keep the original global generic alive and correlate via group_id")
        // with an explicit NSP IR anchor that carries the later destination
        // materialization.
        //
        // We already know this generic has a valid materialize_in_destination
        // sink because `sink` was required above in non-collective mode.
        auto mat = sink->first;
        auto &wrappers = sink->second;

        Value materializeSource =
            rebuildPostSinkTensorChain(localResult, wrappers);
        if (!materializeSource) {
          g.emitError() << "NSPLocalize: failed to rebuild local tensor chain "
                           "before materialization";
          signalPassFailure();
          return;
        }

        Value dest = mat.getOperand(1);
        auto destTy = dyn_cast<MemRefType>(dest.getType());
        if (!destTy) {
          g.emitError() << "NSPLocalize: destination of "
                           "materialize_in_destination is not "
                           "a memref";
          signalPassFailure();
          return;
        }

        auto splitAxisAttr = localizedGeneric->getAttrOfType<IntegerAttr>(
            "nsp.materialize_split_axis");

        auto materializeSourceTy =
            dyn_cast<RankedTensorType>(materializeSource.getType());

        if (!materializeSourceTy || !splitAxisAttr) {
          localizedGeneric.emitError() << "NSPLocalize: localized generic is "
                                          "missing tile materialization "
                                          "attributes";
          signalPassFailure();
          return;
        }

        SmallVector<Attribute> tileShapeElems;
        tileShapeElems.reserve(materializeSourceTy.getRank());
        for (int64_t dim : materializeSourceTy.getShape()) {
          if (ShapedType::isDynamic(dim) || dim <= 0) {
            localizedGeneric.emitError()
                << "NSPLocalize: materialized local tile has dynamic shape";
            signalPassFailure();
            return;
          }
          tileShapeElems.push_back(b.getI64IntegerAttr(dim));
        }

        auto tileShapeArrayAttr = b.getArrayAttr(tileShapeElems);

        // Emit the explicit hand-off op for NSPMaterializePass.
        b.setInsertionPoint(mat);
        mlir::hexagon::nsp::MaterializeTileOp::create(
            b, loc,
            /*source=*/materializeSource,
            /*dest=*/dest,
            /*grid=*/SymbolRefAttr::get(ctx, grid.getSymName()),
            /*splitAxis=*/splitAxisAttr,
            /*tileShape=*/tileShapeArrayAttr);

        // The old sink chain is now replaced by the explicit NSP
        // materialization anchor.
        //
        // Only erase operations that became dead. In blocked_layer_norm, the
        // same sharded value may feed both:
        //   1. tensor.expand_shape -> materialize_in_destination
        //   2. downstream linalg.generic consumers
        //
        // In that case, the side materialization path should disappear, but
        // the global producer/wrapper chain must remain available for the
        // downstream computation.
        mat.erase();
        for (Operation *op : llvm::reverse(wrappers))
          if (op->use_empty())
            op->erase();

        if (g->use_empty())
          g.erase();

        continue;

      } // for worklist
    }); // module.walk

    // Final cleanup
    stripShardAnnotationWrappers(module);

  } // runOnOperation

private:
};

} // namespace

std::unique_ptr<Pass> createNSPLocalizePass() {
  return std::make_unique<NSPLocalizePass>();
}

std::unique_ptr<Pass> createNSPLocalizePass(bool allowCollectives) {
  return std::make_unique<NSPLocalizePass>(allowCollectives);
}

} // namespace hexagon
} // namespace mlir
