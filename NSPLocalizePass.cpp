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
// (e.g. vadd expressed as linalg.generic) for ranked tensors of rank 1 or 2
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
    outLocalInit = b.create<tensor::EmptyOp>(loc, localTy.getShape(),
                                             localTy.getElementType());
  }

  auto localizedGeneric =
      b.create<linalg::GenericOp>(loc,
                                  /*resultTensorTypes=*/TypeRange{localTy},
                                  /*inputs=*/ValueRange{localInputs},
                                  /*outputs=*/ValueRange{outLocalInit},
                                  /*indexingMaps=*/g.getIndexingMaps(),
                                  /*iteratorTypes=*/g.getIteratorTypes(),
                                  /*doc=*/nullptr, /*libraryCall=*/nullptr);

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

/// Return true iff `g` matches the following elementwise pattern:
///   - rank-2 result
///   - all-parallel iterators
///   - output map is identity 2D
///   - each input is either:
///       * rank-2 with identity 2D map, or
///       * rank-1 with row-broadcast map (d0, d1) -> (d0)
static bool
isSupportedBroadcastRowWiseGeneric(linalg::GenericOp g,
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
      if (!isRowBroadcastMap(map))
        return false;
      continue;
    }

    return false;
  }

  return true;
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

    // Helper for strip shard tensor annotations inside loops that were
    // explicitly distributed by this pass (marked with 'nsp.distributed').
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
            if (!forOp->hasAttr("nsp.distributed"))
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

      // Returns true iff the loop appears to perform per-iteration output
      // writes whose address depends on the IV, i.e. a DOALL-style tiled store.
      auto shouldDistributeLoop = [&](mlir::scf::ForOp forOp) -> bool {
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
              if (auto v = dyn_cast<Value>(ofr)) {
                if (reachesFromIv(iv, v)) {
                  foundIvIndexedStore = true;
                  return WalkResult::interrupt();
                }
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
        Value procIdx = b.create<mlir::shard::ProcessLinearIndexOp>(loc, grid);
        // Cast procIdx to the IV type (index stays index; integer gets
        // index_cast).
        Value procInIvTy = procIdx;
        if (!ivIsIndex)
          procInIvTy = b.create<arith::IndexCastOp>(loc, ivTy, procIdx);

        // newLb = lb + procI32 * step
        Value offset = b.create<arith::MulIOp>(loc, procInIvTy, step);
        Value newLb = b.create<arith::AddIOp>(loc, lb, offset);

        // newStep = step * numShards
        Value cNum = ivIsIndex
                         ? static_cast<Value>(
                               b.create<arith::ConstantIndexOp>(loc, numShards))
                         : static_cast<Value>(b.create<arith::ConstantIntOp>(
                               loc, numShards, ivIntTy.getWidth()));

        Value newStep = b.create<arith::MulIOp>(loc, step, cNum);

        // Create the distributed loop.
        auto newFor = b.create<mlir::scf::ForOp>(loc, newLb, ub, newStep);
        // Mark this loop so we can precisely clean up stale shard annotations
        // only within distributed loops (non-collective mode).
        newFor->setAttr("nsp.distributed", b.getUnitAttr());

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
        b.create<mlir::scf::YieldOp>(loc);

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
    //   - Only rank-1 / rank-2 ranked tensors.
    //   - Only all-parallel iterator spaces.
    //   - Only sharding split along axis 0.
    //   - Only ranked tensors with static dim0 divisible by the grid size.
    const SmallVector<mlir::shard::GridAxis> gridAxes = {
        static_cast<mlir::shard::GridAxis>(0)};
    // Some Shard ops (e.g. all_gather) expect grid_axes as i16.
    const SmallVector<int16_t> gridAxesI16 = {static_cast<int16_t>(0)};

    const int64_t splitAxis = 0;
    const llvm::APInt splitAxisAP(/*numBits=*/64, /*val=*/splitAxis,
                                  /*isSigned=*/true);

    // Helper to compute a "local" tensor type for split along axis 0.
    auto getLocalType = [&](RankedTensorType globalTy) -> RankedTensorType {
      if (!globalTy || globalTy.getRank() == 0)
        return RankedTensorType();
      int64_t dim0 = globalTy.getDimSize(0);
      if (ShapedType::isDynamic(dim0))
        return RankedTensorType();
      if (dim0 % numShards != 0)
        return RankedTensorType();
      SmallVector<int64_t> newShape(globalTy.getShape().begin(),
                                    globalTy.getShape().end());
      newShape[0] = dim0 / numShards;
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
        [&](RankedTensorType inTy, RankedTensorType outTy) -> bool {
      if (!inTy || !outTy || outTy.getRank() != 2)
        return false;

      auto outLocalTy = getLocalType(outTy);
      if (!outLocalTy)
        return false;

      if (inTy.getRank() == 2) {
        auto inLocalTy = getLocalType(inTy);
        if (!inLocalTy)
          return false;
        return inLocalTy.getShape() == outLocalTy.getShape();
      }

      if (inTy.getRank() == 1) {
        auto inLocalTy = getLocalType(inTy);
        if (!inLocalTy)
          return false;
        return inLocalTy.getShape().size() == 1 &&
               outLocalTy.getShape().size() == 2 &&
               inLocalTy.getShape()[0] == outLocalTy.getShape()[0];
      }

      return false;
    };

    // Compute the shard-local tensor type expected for a specific operand of
    // a localized generic.
    //
    // For the current broadcast-aware path that:
    //   - rank-2 identity inputs use the same local type as the result
    //   - rank-1 row-broadcast inputs ((d0, d1) -> (d0)) use only the local
    //   row extent, i.e. tensor<localRows x T> -> tensor<localRows x T>
    //   becomes tensor<localRows x elemTy> for the rank-1 operand
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

      return RankedTensorType();
    };

    auto isSupportedElementwiseGenericShape =
        [&](mlir::linalg::GenericOp g, RankedTensorType outResTy,
            ArrayRef<RankedTensorType> inputTys) -> bool {
      if (!outResTy)
        return false;

      const int64_t rank = outResTy.getRank();
      if (rank != 1 && rank != 2)
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

    module.walk([&](mlir::func::FuncOp func) {
      OpBuilder b(func.getContext());

      // Find a bufferization.materialize_in_destination sink for `v` while
      // allowing a trivial chain of sharding wrappers.
      //
      // Sharding propagation often wraps values with shard.shard, so the IR can
      // look like:
      //   %r  = linalg.generic ... -> tensor<...>
      //   %r1 = shard.shard %r to %sharding
      //   bufferization.materialize_in_destination %r1 in %dst
      //
      // In store-by-tile mode, we rewrite the materialization to store the
      // local tile into a subview of %dst, and erase the wrapper chain.
      auto findMaterializeSink = [&](Value v)
          -> std::optional<std::pair<bufferization::MaterializeInDestinationOp,
                                     SmallVector<Operation *>>> {
        SmallVector<Operation *> wrappers;
        Value cur = v;

        while (true) {
          // Keep bring-up simple: require a single-use chain.
          if (!cur.hasOneUse())
            return std::nullopt;

          Operation *user = *cur.getUsers().begin();

          // Allow shard.shard wrappers.
          if (auto shardOp = dyn_cast<mlir::shard::ShardOp>(user)) {
            // Be robust to ShardOp operand naming differences across versions.
            if (shardOp->getNumOperands() < 1 || shardOp->getOperand(0) != cur)
              return std::nullopt;
            wrappers.push_back(user);
            cur = shardOp->getResult(0);
            continue;
          }

          // Allow an optional tensor.cast in between.
          if (auto castOp = dyn_cast<tensor::CastOp>(user)) {
            if (castOp.getSource() != cur)
              return std::nullopt;
            wrappers.push_back(user);
            cur = castOp.getResult();
            continue;
          }

          // Accept bufferization.materialize_in_destination.
          if (auto mat =
                  dyn_cast<bufferization::MaterializeInDestinationOp>(user)) {
            if (mat->getOperand(0) != cur)
              return std::nullopt;
            return std::make_optional(std::make_pair(mat, wrappers));
          }

          return std::nullopt;
        }
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

      auto isCompatibleElementwiseGeneric =
          [&](mlir::linalg::GenericOp prod,
              RankedTensorType expectedLocalTy) -> bool {
        if (!expectedLocalTy)
          return false;

        const int64_t expectedRank = expectedLocalTy.getRank();
        if (expectedRank != 1 && expectedRank != 2)
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

      // Build a local-tile version of a global value by cloning a chain of
      // compatible elementwise producers. This avoids materializing full global
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

        // linalg.fill: produce a local constant tile.
        if (auto fill =
                dyn_cast_or_null<mlir::linalg::FillOp>(base.getDefiningOp())) {
          Value outLocalInit = b.create<mlir::tensor::EmptyOp>(
              fill.getLoc(), expectedLocalTy.getShape(),
              expectedLocalTy.getElementType());
          auto localFill = b.create<mlir::linalg::FillOp>(
              fill.getLoc(), /*inputs=*/fill.getInputs(),
              /*outputs=*/ValueRange{outLocalInit});
          Value localRes = localFill.getResult(0);
          cache[base] = localRes;
          return localRes;
        }

        // linalg.generic: clone producer locally if compatible.
        if (auto prod = dyn_cast_or_null<mlir::linalg::GenericOp>(
                base.getDefiningOp())) {
          if (isCompatibleElementwiseGeneric(prod, expectedLocalTy)) {
            SmallVector<Value> prodInputs;
            const int64_t nIn = prod.getNumDpsInputs();
            prodInputs.reserve(nIn);
            for (int64_t i = 0; i < nIn; ++i) {
              Value inV = prod.getDpsInputOperand(i)->get();
              prodInputs.push_back(
                  materializeLocalValue(inV, expectedLocalTy, cache));
            }

            Value outLocalInit = b.create<mlir::tensor::EmptyOp>(
                prod.getLoc(), expectedLocalTy.getShape(),
                expectedLocalTy.getElementType());

            auto newProd = b.create<mlir::linalg::GenericOp>(
                prod.getLoc(),
                /*resultTensorTypes=*/TypeRange{expectedLocalTy},
                /*inputs=*/ValueRange{prodInputs},
                /*outputs=*/ValueRange{outLocalInit},
                /*indexingMaps=*/prod.getIndexingMaps(),
                /*iteratorTypes=*/prod.getIteratorTypes(),
                /*doc=*/nullptr, /*libraryCall=*/nullptr);

            // Ensure the region is empty before cloning.
            newProd.getRegion().getBlocks().clear();

            // Clone the region body.
            Block &srcBlock = prod.getRegion().front();
            Block *dstBlock = new Block();
            newProd.getRegion().push_back(dstBlock);
            for (BlockArgument a : srcBlock.getArguments())
              dstBlock->addArgument(a.getType(), a.getLoc());

            IRMapping map;
            for (auto [sa, da] :
                 llvm::zip(srcBlock.getArguments(), dstBlock->getArguments()))
              map.map(sa, da);

            OpBuilder nb = OpBuilder::atBlockEnd(dstBlock);
            for (Operation &op : srcBlock.getOperations())
              nb.clone(op, map);

            Value localRes = newProd.getResult(0);
            cache[base] = localRes;
            return localRes;
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

      // Use a worklist since we'll rewrite in-place.
      SmallVector<mlir::linalg::GenericOp> worklist;
      func.walk([&](mlir::linalg::GenericOp g) { worklist.push_back(g); });

      for (mlir::linalg::GenericOp g : worklist) {

        // Initial structural filter.
        const int64_t numInputs = g.getNumDpsInputs();
        if (numInputs < 1 || numInputs > 3 || g.getNumDpsInits() != 1)
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
        bool supportsBroadcastRowWise =
            isSupportedBroadcastRowWiseGeneric(g, outResTy, inputTys);

        if (!supportsIdentityElementwise && !supportsBroadcastRowWise)
          continue;

        auto localTy = getLocalType(outResTy);
        if (!localTy) {
          g.emitError() << "NSPLocalize: cannot compute local tile type for "
                           "result "
                        << outResTy << " with grid size " << numShards
                        << " (requires static dim0 divisible by grid size)";
          signalPassFailure();
          return;
        }

        // Inputs must shard to the same local shape as the result.
        // For broadcast-aware rank-2 generics, rank-1 inputs are considered
        // compatible when their shard-local extent matches the row dimension
        // of the output tile.
        bool compatibleLocalShapes = true;
        for (RankedTensorType t : inputTys) {
          if (supportsBroadcastRowWise) {
            if (!isCompatibleInputForBroadcastLocalize(t, outResTy)) {
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
        if (!allowCollectives)
          if (auto parentFor = g->getParentOfType<mlir::scf::ForOp>())
            if (parentFor->hasAttr("nsp.distributed"))
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
          RankedTensorType expectedInputLocalTy = localTy;

          if (supportsBroadcastRowWise) {
            expectedInputLocalTy =
                getExpectedLocalTypeForOperand(inputTys[i], maps[i], localTy);
            if (!expectedInputLocalTy) {
              g.emitError()
                  << "NSPLocalize: cannot compute operand-local type for input "
                  << i << " of broadcast-aware generic";
              signalPassFailure();
              return;
            }
          }

          Value localIn = materializeLocalValue(inputs[i], expectedInputLocalTy,
                                                localCache);
          if (!localIn)
            continue;
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

        if (allowCollectives) {
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

        Value dest = mat.getOperand(1);
        auto destTy = dyn_cast<MemRefType>(dest.getType());
        if (!destTy) {
          g.emitError() << "NSPLocalize: destination of "
                           "materialize_in_destination is not "
                           "a memref";
          signalPassFailure();
          return;
        }

        auto tileShapeDenseAttr =
            localizedGeneric->getAttrOfType<DenseI64ArrayAttr>(
                "nsp.materialize_tile_shape");
        auto splitAxisAttr = localizedGeneric->getAttrOfType<IntegerAttr>(
            "nsp.materialize_split_axis");

        if (!tileShapeDenseAttr || !splitAxisAttr) {
          localizedGeneric.emitError() << "NSPLocalize: localized generic is "
                                          "missing tile materialization "
                                          "attributes";
          signalPassFailure();
          return;
        }

        SmallVector<Attribute> tileShapeElems;
        tileShapeElems.reserve(tileShapeDenseAttr.size());
        for (int64_t dim : tileShapeDenseAttr.asArrayRef())
          tileShapeElems.push_back(b.getI64IntegerAttr(dim));

        auto tileShapeArrayAttr = b.getArrayAttr(tileShapeElems);

        // Emit the explicit hand-off op for NSPMaterializePass.
        b.setInsertionPoint(mat);
        b.create<mlir::hexagon::nsp::MaterializeTileOp>(
            loc,
            /*source=*/localResult,
            /*dest=*/dest,
            /*grid=*/SymbolRefAttr::get(ctx, grid.getSymName()),
            /*splitAxis=*/splitAxisAttr,
            /*tileShape=*/tileShapeArrayAttr);

        // The old sink chain is now replaced by the explicit NSP
        // materialization anchor, so the old sink/wrappers/global generic can
        // go away now.
        mat.erase();
        for (Operation *op : llvm::reverse(wrappers))
          op->erase();
        g.erase();

        continue;

      } // for worklist
    }); // module.walk
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
