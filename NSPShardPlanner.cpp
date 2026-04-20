//===--- NSPShardPlanner.cpp - implement a basic shard planner for NSP  ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// This file implements a sharding planner for NSP SPMD execution.
//
// The pass inspects linalg.generic operations and attaches shard dialect
// annotations based on iterator_types and indexing_maps. It optionally
// materializes communication (e.g. all-reduce) when the chosen sharding
// implies cross-NSP reductions.
//
// Design goals:
//   * Work at the "linalg.generic" level (pre-tiling).
//   * Be dialect-agnostic except for using the shard dialect as the sharding
//   IR.
//   * Keep the pipeline compatible with further lowering to per-NSP slices
//     (tensor.extract_slice/memref.subview) and runtime intrinsics.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Shard/IR/ShardDialect.h"
#include "mlir/Dialect/Shard/IR/ShardOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Transforms/OptionsParsing.h"

#define DEBUG_TYPE "nsp-splanner"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::linalg;
using namespace hexagon;

namespace {

/// A simple policy describing how we want to shard a computation.
/// In practice we may want this to be configurable (via pass options),
/// or drive it from a cost model (e.g. memory traffic vs compute).
struct ShardPolicy {
  // The number of NSPs in the grid (1-D mesh for now).
  int64_t nspCount = 16;

  // Prefer sharding along the first parallel iterator that indexes the output.
  bool preferFirstOutputParallelDim = true;

  // If true, allow sharding decisions that introduce collectives (e.g.
  // all-reduce). For initial release "no inter-NSP traffic" mode, so we keep
  // this false.
  bool allowCollectives = false;
};

/// A structured description of the sharding decision for a single op.
struct ShardPlan {
  // The chosen loop iterator index to shard (e.g. i or j in matmul).
  int64_t shardIter = -1;

  // Whether this plan requires a cross-NSP all-reduce to be correct.
  bool requiresAllReduce = false;

  // The shard axis on the device grid (only 0 for 1-D grid).
  int64_t gridAxis = 0;
};

/// Planner pass that attaches explicit shard annotations to linalg.generic ops.
///
/// This pass analyzes linalg::GenericOp operations and materializes a
/// declarative sharding plan using the shard dialect. The result is an IR
/// where tensor operands are annotated with explicit `!shard.sharding` values
/// and wrapped by `shard.shard` ops, describing how data is partitioned across
/// a logical shard grid.
///
/// This pass produces a first-class, SSA-based sharding IR that can
/// be reasoned about, propagated, and refined by downstream passes.
///
/// Sharding decisions made by this pass are conservative and operand-local:
///  - A tensor dimension is sharded only if it is indexed directly by a
///    selected loop iterator.
///  - All other dimensions are treated as replicated.
///  - The current implementation assumes a 1-D shard grid.
///
/// This separation of concerns keeps planning, propagation, and lowering
/// orthogonal, enabling multiple SPMDization strategies to consume the same
/// declarative sharding IR.
struct NSPShardPlannerPass
    : public PassWrapper<NSPShardPlannerPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NSPShardPlannerPass)

  // Ensure Shard dialect is loaded
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::shard::ShardDialect>();
  }

  // Pass command-line identifier
  StringRef getArgument() const override { return "nsp-shard-planner"; }

  // Short help text for --list-passes and --help.
  StringRef getDescription() const override {
    return "Plan Shard dialect annotations for multi-NSP (pre-tiling)";
  }

  NSPShardPlannerPass() = default;
  NSPShardPlannerPass(const ShardPolicy &policy) : policy(policy) {}

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ctx->getOrLoadDialect<mlir::shard::ShardDialect>();

    // Ensure we have a shard.grid symbol available in the module.
    func::FuncOp func = getOperation();
    ModuleOp module = func->getParentOfType<ModuleOp>();
    shard::GridOp grid = getOrCreateGrid(module, policy.nspCount);

    llvm::StringMap<Value> shardingCache;

    // We rebuild (and erase) linalg.generic ops during annotation.
    // Erasing ops while walking can invalidate the walk. We collect
    // first, then rewrite in a stable loop.
    SmallVector<linalg::GenericOp> generics;

    func.walk([&](linalg::GenericOp op) {
      linalg::LinalgOp linalgOp(op);
      // Only handle ops with ranked tensor semantics (needs pre-bufferization).
      if (linalgOp.hasPureTensorSemantics())
        generics.push_back(op);
    });

    for (linalg::GenericOp op : generics) {
      // The op may have been erased by a previous rewrite.
      if (!op || !op->getParentOp())
        continue;

      // Analyze the op and build a sharding plan.
      FailureOr<ShardPlan> plan = buildPlanForGeneric(op, policy);
      if (failed(plan))
        continue;

      // If the plan introduces collectives but they are not allowed, bail out.
      if (plan->requiresAllReduce && !policy.allowCollectives)
        continue;

      // Attach shard annotations to operands using shard.shard AND rebuild
      // the linalg.generic so that it consumes the annotated values.
      linalg::GenericOp newOp =
          annotateGenericWithShard(op, grid, *plan, shardingCache);
      // If needed, insert shard collectives (e.g. all-reduce) explicitly.
      // In many pipelines, a later pass does this; but we show it here
      // to make the semantics explicit.
      if (plan->requiresAllReduce)
        materializeAllReduceIfNeeded(newOp, *plan);
    }
  }

private:
  ShardPolicy policy;

  /// Get or create `shard.grid @nsp(shape = <nspCount>)` in the module.
  static shard::GridOp getOrCreateGrid(ModuleOp module, int64_t nspCount) {
    MLIRContext *ctx = module.getContext();
    OpBuilder b(ctx);

    // Look up a pre-existing grid named "nsp".
    if (auto grid = module.lookupSymbol<shard::GridOp>("nsp"))
      return grid;

    b.setInsertionPointToStart(module.getBody());
    SmallVector<int64_t, 1> shape = {nspCount};
    return b.create<shard::GridOp>(module.getLoc(), "nsp",
                                   llvm::ArrayRef<int64_t>(shape));
  }

  /// Build a stable cache key for a sharding configuration.
  /// The key is derived from the grid symbol and per-dimension split axes.
  static std::string
  buildShardingCacheKey(shard::GridOp grid,
                        ArrayRef<shard::GridAxesAttr> splitAxes) {
    std::string key;
    llvm::raw_string_ostream os(key);

    os << grid.getSymName();
    os << '|';

    for (shard::GridAxesAttr axes : splitAxes) {
      os << '[';
      auto ref = axes.asArrayRef();
      for (size_t i = 0; i < ref.size(); ++i) {
        if (i)
          os << ',';
        os << ref[i];
      }
      os << ']';
    }

    return os.str();
  }

  /// Return an existing !shard.sharding value for the same grid/split_axes,
  /// or create and cache a new one if it does not exist yet.
  static Value getOrCreateShardingValue(OpBuilder &b, Location loc,
                                        shard::GridOp grid,
                                        ArrayRef<shard::GridAxesAttr> splitAxes,
                                        llvm::StringMap<Value> &shardingCache) {
    std::string key = buildShardingCacheKey(grid, splitAxes);

    if (auto it = shardingCache.find(key); it != shardingCache.end())
      return it->second;

    auto gridRef = FlatSymbolRefAttr::get(grid.getSymNameAttr());

    auto shardingOp = b.create<shard::ShardingOp>(
        loc,
        /*grid=*/gridRef,
        /*split_axes=*/splitAxes,
        /*static_halo_sizes=*/ArrayRef<int64_t>{},
        /*static_sharded_dims_offsets=*/ArrayRef<int64_t>{});

    Value result = shardingOp.getResult();
    shardingCache.try_emplace(key, result);
    return result;
  }

  /// Return true iff `map` is compatible with sharding loop iterator
  /// `shardIter`.
  ///
  /// Rationale:
  ///   The NSP planner chooses a single loop iterator per linalg.generic to
  ///   drive the sharding decision. However, different operands of the same op
  ///   may legitimately have different ranks and different projected views of
  ///   that iterator.
  ///
  ///   Example:
  ///     ins(tensor<MxN>, tensor<M>) -> outs(tensor<MxN>)
  ///     maps:
  ///       (d0, d1) -> (d0, d1)
  ///       (d0, d1) -> (d0)
  ///       (d0, d1) -> (d0, d1)
  ///
  ///   If the chosen shard iterator is d0, then the matrix operands/results
  ///   should receive split_axes like [[0], []], while the rank-1 broadcast
  ///   operand should receive [[0]].
  ///
  ///   This is semantically valid: all shardings are derived from the same
  ///   loop iterator, even though the final split_axes arrays differ in rank
  ///   and structure.
  ///
  /// Current conservative rule:
  ///   - require the indexing map to be a projected permutation of loop dims
  ///   - require every result expr to be a direct AffineDimExpr
  ///
  /// This is intentionally simple and sufficient for current bring-up
  /// elementwise/broadcast patterns such as online softmax.
  static bool isOperandMapCompatibleWithShardIter(AffineMap map,
                                                  int64_t shardIter) {
    (void)shardIter;

    if (!map.isProjectedPermutation())
      return false;

    for (AffineExpr e : map.getResults()) {
      if (!dyn_cast<AffineDimExpr>(e))
        return false;
    }

    return true;
  }

  /// Return true iff the given sharding value matches the sharding that would
  /// be inferred locally for (`map`, `tensorVal`).
  ///
  /// This is used to avoid reusing a previously attached shard.shard when the
  /// same SSA value flows into multiple consumers with different indexing maps.
  ///
  /// Example:
  ///   A rank-1 tensor may be sharded as [[0]] when consumed via
  ///   (d0, d1) -> (d0), but a blindly reused sharding from a different
  ///   consumer would be incorrect if it came from a different map.
  static bool shardingMatchesMap(
      Value shardingVal, AffineMap map, Value tensorVal,
      function_ref<Value(AffineMap, Value)> buildShardingValueFor) {
    if (!shardingVal)
      return false;

    Value expected = buildShardingValueFor(map, tensorVal);
    if (!expected)
      return false;

    return expected == shardingVal;
  }

  /// Look for an existing shard.shard that already annotates `src` with
  /// the given sharding value, and return its result if found.
  /// Obs.: The reuse is valid only for inputs.
  static Value getExistingShardFor(Value src, Value sharding) {
    for (Operation *user : src.getUsers()) {
      auto shardOp = dyn_cast<shard::ShardOp>(user);
      if (!shardOp)
        continue;
      if (shardOp.getSrc() == src && shardOp.getSharding() == sharding)
        return shardOp.getResult();
    }
    return Value();
  }

  /// Reuse an existing shard.shard for (`src`, `sharding`) when available;
  /// otherwise create a new shard annotation op. obs.: input only
  static Value getOrCreateShardFor(OpBuilder &b, Location loc, MLIRContext *ctx,
                                   Value src, Value sharding) {

    if (Value existing = getExistingShardFor(src, sharding))
      return existing;

    auto shardOp =
        b.create<shard::ShardOp>(loc, src.getType(),
                                 /*src=*/src,
                                 /*sharding=*/sharding,
                                 /*annotate_for_users=*/UnitAttr::get(ctx));

    return shardOp.getResult();
  }

  /// Prefer an existing sharding attached to `value`; otherwise build a new
  /// sharding descriptor from the current operand indexing map.
  static Value getOrInferInputSharding(
      OpBuilder &b, Location loc, Value value, AffineMap map,
      function_ref<Value(AffineMap, Value)> buildShardingValueFor) {
    (void)b;
    (void)loc;

    // Fast path: the value is directly produced by shard.shard.
    if (auto shardOp = value.getDefiningOp<shard::ShardOp>()) {
      if (shardingMatchesMap(shardOp.getSharding(), map, value,
                             buildShardingValueFor))
        return shardOp.getSharding();
    }

    // Fallback: If the value already participates in a shard
    // annotation that matches this consumer map, reuse it instead of
    // recomputing locally.
    //
    // Important:
    //   We do NOT blindly reuse any pre-existing sharding attached to the
    //   value. The same producer may feed different consumers with different
    //   indexing maps (e.g. broadcast/projected consumers), so the consumer's
    //   local map must agree with the reused sharding.
    for (Operation *user : value.getUsers()) {
      if (auto shardOp = dyn_cast<shard::ShardOp>(user)) {
        if (shardOp.getSrc() == value &&
            shardingMatchesMap(shardOp.getSharding(), map, value,
                               buildShardingValueFor))
          return shardOp.getSharding();
      }
    }

    // Otherwise, infer sharding from the local consumer map.
    return buildShardingValueFor(map, value);
  }

  /// If `value` is produced by linalg.transpose, map a source tensor dimension
  /// `srcDim` to the corresponding result dimension. Return std::nullopt if
  /// the mapping cannot be derived.
  static std::optional<int64_t>
  mapTransposeSourceDimToResultDim(Value value, int64_t srcDim) {
    auto transposeOp = value.getDefiningOp<linalg::TransposeOp>();
    if (!transposeOp)
      return std::nullopt;

    auto permutation = transposeOp.getPermutation();
    for (auto it : llvm::enumerate(permutation)) {
      if (it.value() == srcDim)
        return static_cast<int64_t>(it.index());
    }
    return std::nullopt;
  }

  /// Try to choose a shard iterator that is consistent with a transpose input.
  /// Heuristic:
  ///   - If the generic has a transpose input
  ///   - and the output map is an identity/projected permutation
  ///   - then preserve the canonical "source dim 0" sharding through the
  ///     transpose permutation.
  /// This is especially useful for softmax-like patterns where a tensor is
  /// transposed before a pointwise op and later reduced.
  static std::optional<int64_t>
  pickShardIteratorFromTransposeInput(linalg::GenericOp op) {
    auto maps = op.getIndexingMapsArray();
    linalg::LinalgOp linalgOp(op);

    if (linalgOp.getNumDpsInits() != 1)
      return std::nullopt;

    AffineMap outMap = maps[linalgOp.getNumDpsInputs()];
    if (!outMap.isProjectedPermutation())
      return std::nullopt;

    for (unsigned i = 0; i < linalgOp.getNumDpsInputs(); ++i) {
      Value in = op.getInputs()[i];
      auto maybeResultDim = mapTransposeSourceDimToResultDim(in, /*srcDim=*/0);
      if (!maybeResultDim)
        continue;

      int64_t resultDim = *maybeResultDim;
      if (resultDim < 0 ||
          resultDim >= static_cast<int64_t>(outMap.getNumResults()))
        continue;

      if (auto dimExpr = dyn_cast<AffineDimExpr>(outMap.getResult(resultDim)))
        return static_cast<int64_t>(dimExpr.getPosition());
    }

    return std::nullopt;
  }

  /// Build a sharding plan for a linalg.generic op.
  ///
  /// The key inputs are:
  ///   * iterator_types: parallel vs reduction
  ///   * indexing_maps: which loop dims index which operands/results
  ///
  /// Heuristic:
  ///   * Prefer a parallel iterator that appears as a direct dim in the output
  ///   map.
  ///   * If sharding would shard a reduced dimension that must be aggregated,
  ///     mark requiresAllReduce=true.
  static FailureOr<ShardPlan> buildPlanForGeneric(linalg::GenericOp op,
                                                  const ShardPolicy &policy) {
    ShardPlan plan;

    // Extract iterator types
    SmallVector<mlir::utils::IteratorType> iters;
    iters.reserve(op.getNumLoops());
    for (mlir::utils::IteratorType it : op.getIteratorTypesArray())
      iters.push_back(it);

    // Identify output indexing map
    // We assume single output for simplicity (generalize as needed).
    linalg::LinalgOp linalgOp(op);
    if (linalgOp.getNumDpsInits() != 1)
      return failure();
    AffineMap outMap = op.getIndexingMapsArray().back();

    // Pick a sharding iterator.
    if (auto transposeChoice = pickShardIteratorFromTransposeInput(op))
      plan.shardIter = *transposeChoice;
    else
      plan.shardIter = pickShardIteratorIndex(outMap, iters, policy);
    if (plan.shardIter < 0)
      return failure();

    // Validate that every operand/result map is compatible with deriving
    // operand-local split_axes from the single chosen shard iterator.
    //
    // This intentionally allows heterogenous operand shardings, including
    // different ranks, as long as they are all consistent with the same
    // loop iterator decision. This is required for broadcast-style patterns
    // such as:
    //   ins(tensor<MxN>, tensor<M>) -> outs(tensor<MxN>)
    for (AffineMap map : op.getIndexingMapsArray()) {
      if (!isOperandMapCompatibleWithShardIter(map, plan.shardIter))
        return failure();
    }

    // Determine if this sharding requires a cross-NSP all-reduce.
    //
    // If the shardIter corresponds to a reduction-only dimension of the final
    // value, then each NSP computes partial results that need aggregation.
    //
    // For elementwise ops: false.
    // For matmul sharded on i or j: false (k is local reduction).
    // For global sum sharded on reduced axis: true.
    plan.requiresAllReduce =
        shardingIntroducesGlobalReduction(op, plan.shardIter);

    return plan;
  }

  /// Decide which iterator to shard based on output map and iterator_types.
  /// \returns the loop iterator index, or -1 if no suitable iterator exists.
  static int64_t
  pickShardIteratorIndex(AffineMap outMap,
                         ArrayRef<mlir::utils::IteratorType> iteratorTypes,
                         const ShardPolicy &policy) {

    // We want an iterator that:
    //   (1) is "parallel"
    //   (2) appears as a plain AffineDimExpr in the output map results.
    //
    // Example: outMap = (i,j,k) -> (i,j)
    // candidates: i and j (if both are parallel iterators).
    for (AffineExpr e : outMap.getResults()) {
      auto dim = dyn_cast<AffineDimExpr>(e);
      if (!dim)
        continue;
      int64_t iterIdx = dim.getPosition();
      if (iterIdx < 0 || iterIdx >= (int64_t)iteratorTypes.size())
        continue;
      if (iteratorTypes[iterIdx] == mlir::utils::IteratorType::parallel)
        return iterIdx;
    }
    return -1;
  }

  /// Return the set of loop iterator indices used by an affine map result list.
  /// We only consider direct AffineDimExpr (no affine.apply or arithmetic),
  /// keeping this intentionally conservative.
  static llvm::SmallDenseSet<int64_t> collectLoopItersUsedByMap(AffineMap map) {
    llvm::SmallDenseSet<int64_t> used;
    for (AffineExpr e : map.getResults()) {
      if (auto d = dyn_cast<AffineDimExpr>(e))
        used.insert(d.getPosition());
    }
    return used;
  }

  /// Return true if the map is a pure projected permutation from loop dims to
  /// tensor dims (i.e. results are distinct AffineDimExpr).
  static bool isProjectedPermutationOfDims(AffineMap map) {
    // MLIR already provides a check.
    // This will be false if there is any arithmetic / symbols / repeated dims.
    return map.isProjectedPermutation();
  }

  /// Recognize the canonical 2D matmul contraction pattern:
  ///   iterators: [parallel, parallel, reduction]
  ///   maps: A(i,k), B(k,j), C(i,j)
  ///
  /// This is meant to catch the common linalg.generic that came from
  /// matmul-like lowering (or hand-written).
  static bool matchMatmulLike(linalg::GenericOp op, int64_t &iIter,
                              int64_t &jIter, int64_t &kIter) {
    auto iters = op.getIteratorTypesArray();
    if (iters.size() < 3)
      return false;

    // Collect parallel and reduction iter indices.
    SmallVector<int64_t> parallelIters;
    SmallVector<int64_t> reductionIters;
    for (int64_t idx = 0; idx < (int64_t)iters.size(); ++idx) {
      if (iters[idx] == utils::IteratorType::parallel)
        parallelIters.push_back(idx);
      else if (iters[idx] == utils::IteratorType::reduction)
        reductionIters.push_back(idx);
    }

    // Canonical matmul has exactly 2 parallel + 1 reduction.
    // If the IR has extra loops, we relax this, but keep it conservative.
    if (parallelIters.size() != 2 || reductionIters.size() != 1)
      return false;

    // Wrap the generic op with the LinalgOp interface to access common Linalg
    // helpers.
    linalg::LinalgOp linalgOp(op);

    // Must have at least 2 inputs + 1 output for matmul-like.
    if (linalgOp.getNumDpsInputs() < 2 || linalgOp.getNumDpsInits() < 1)
      return false;

    auto maps = op.getIndexingMapsArray();
    AffineMap aMap = maps[0];
    AffineMap bMap = maps[1];
    AffineMap cMap = maps[linalgOp.getNumDpsInputs() + 0];

    // Ensure maps are simple permutations/projections.
    if (!isProjectedPermutationOfDims(aMap) ||
        !isProjectedPermutationOfDims(bMap) ||
        !isProjectedPermutationOfDims(cMap))
      return false;

    // Collect iter usage sets.
    auto aUsed = collectLoopItersUsedByMap(aMap);
    auto bUsed = collectLoopItersUsedByMap(bMap);
    auto cUsed = collectLoopItersUsedByMap(cMap);

    // For canonical A(i,k), B(k,j), C(i,j):
    // - C uses both parallel iters (not the reduction iter).
    // - A uses i and k
    // - B uses k and j
    int64_t p0 = parallelIters[0];
    int64_t p1 = parallelIters[1];
    int64_t r0 = reductionIters[0];

    // C must use both parallel and must not use reduction.
    if (!(cUsed.contains(p0) && cUsed.contains(p1)))
      return false;
    if (cUsed.contains(r0))
      return false;

    // A must use (one parallel) + (reduction)
    // B must use (other parallel) + (reduction)
    bool aIs_p0_r =
        aUsed.contains(p0) && aUsed.contains(r0) && !aUsed.contains(p1);
    bool aIs_p1_r =
        aUsed.contains(p1) && aUsed.contains(r0) && !aUsed.contains(p0);
    bool bIs_p0_r =
        bUsed.contains(p0) && bUsed.contains(r0) && !bUsed.contains(p1);
    bool bIs_p1_r =
        bUsed.contains(p1) && bUsed.contains(r0) && !bUsed.contains(p0);

    if (aIs_p0_r && bIs_p1_r) {
      iIter = p0;
      jIter = p1;
      kIter = r0;
      return true;
    }
    if (aIs_p1_r && bIs_p0_r) {
      iIter = p1;
      jIter = p0;
      kIter = r0;
      return true;
    }

    return false;
  }

  /// Determine whether sharding on `shardIter` forces an all-reduce for
  /// correctness.
  ///
  /// This is intentionally conservative. A production implementation should
  /// understand:
  ///   * how results are used downstream
  ///   * whether the result is expected to be replicated or sharded
  ///   * whether reductions are local vs global
  static bool shardingIntroducesGlobalReduction(linalg::GenericOp op,
                                                int64_t shardIter) {
    // Scalar Result Heuristic: scalar reductions are typically global.
    // -----------------------
    // If the op returns a scalar (rank-0) and any loop is sharded, it's likely
    // a global reduction -> needs all-reduce.
    if (op.getResultTypes().size() == 1) {
      if (auto rt = dyn_cast<RankedTensorType>(op.getResultTypes()[0])) {
        if (rt.getRank() == 0)
          return true;
      }
    }

    // For typical matmul-like patterns, sharding on i/j does not need
    // all-reduce. We can recognize matmul by iterator types and maps:
    //   iterators: [parallel, parallel, reduction]
    //   maps: A(i,k), B(k,j), C(i,j)
    // If shardIter is one of the parallel iters that index C, return false.

    // Fast-path: recognize matmul-like contraction.
    // ---------
    // Sharding on i or j (the parallel iters that index C) is safe (no
    // all-reduce). Sharding on k (the reduction iter) would require reduction
    // across shards.
    int64_t iIter = -1, jIter = -1, kIter = -1;
    if (matchMatmulLike(op, iIter, jIter, kIter)) {
      if (shardIter == iIter || shardIter == jIter)
        return false;
      if (shardIter == kIter)
        return true;
      // If shardIter is something else (shouldn't happen in strict match), be
      // conservative.
      return true;
    }

    // Wrap the generic op with the LinalgOp interface to access common Linalg
    // helpers.
    linalg::LinalgOp linalgOp(op);

    // Conservative fallback:
    // ---------------------
    // If shardIter is a reduction iterator, it often implies each shard
    // computes a partial reduction that must be combined globally.
    //
    // We make it slightly less trigger-happy by checking whether the output
    // indexing map depends on shardIter. If the output depends on shardIter,
    // then sharding that iter changes which output elements are produced by
    // each shard (more like partitioning the output) and may not require
    // all-reduce.
    auto iters = op.getIteratorTypesArray();
    if (shardIter >= 0 && shardIter < (int64_t)iters.size() &&
        iters[shardIter] == utils::IteratorType::reduction) {
      // Look at the first output map (common case).
      auto maps = op.getIndexingMapsArray();
      if (linalgOp.getNumDpsInits() > 0) {
        AffineMap outMap = maps[linalgOp.getNumDpsInputs() + 0];
        auto outUsed = collectLoopItersUsedByMap(outMap);
        // If the output does not depend on the sharded reduction iter,
        // then shards are computing partial sums for the same output -> needs
        // all-reduce.
        if (!outUsed.contains(shardIter))
          return true;
      }
      // Otherwise, keep conservative: still likely needs all-reduce.
      return true;
    }

    // Another Conservative Heuristic:
    // ------------------------------
    // If the op has any reduction iterator at all, and we shard a parallel
    // iterator that does NOT appear in the output map, we are probably
    // partitioning a reduced dimension (i.e. different shards reduce different
    // slices but to the same output), which requires all-reduce.
    bool hasReduction = llvm::any_of(iters, [](utils::IteratorType t) {
      return t == utils::IteratorType::reduction;
    });

    if (hasReduction && op.getNumDpsInits() > 0) {
      auto maps = op.getIndexingMapsArray();
      AffineMap outMap = maps[linalgOp.getNumDpsInputs() + 0];
      auto outUsed = collectLoopItersUsedByMap(outMap);

      if (!outUsed.contains(shardIter)) {
        // Sharding a loop that does not index the output while reductions exist
        // strongly suggests partial reductions that must be combined.
        return true;
      }
    }

    // Default: assume no global reduction is required.
    // This keeps the planner from inserting collectives too eagerly.
    return false;
  }

  /// Annotate a linalg::GenericOp with shard semantics.
  ///
  /// This function materializes sharding information for each tensor operand
  /// (inputs and outputs) of a linalg::GenericOp by:
  ///
  ///  1. Creating explicit `!shard.sharding` SSA values describing how each
  ///     tensor is partitioned across a shard grid.
  ///  2. Wrapping tensor operands with `shard.shard` ops that attach those
  ///     sharding descriptors to the SSA values.
  ///  3. Rebuilding the linalg::GenericOp so that it consumes the annotated
  ///     operands, since sharding is represented as SSA values rather than
  ///     attributes.
  ///
  /// This function is invoked once per linalg::GenericOp selected by the
  /// NSP shard planner, after a sharding plan has been computed (i.e. a loop
  /// iterator to shard has already been chosen).
  ///
  /// The original linalg::GenericOp is replaced by a new one that consumes
  /// the sharded operands; the region body is moved without cloning.
  ///
  /// Important note on broadcast-style operands:
  ///   Different operands of the same linalg.generic may legitimately receive
  ///   different split_axes encodings, including different ranks, as long as
  ///   all shardings are derived from the same chosen shard iterator.
  ///
  ///   Example:
  ///     ins(tensor<MxN>, tensor<M>) -> outs(tensor<MxN>)
  ///   may produce:
  ///     [[0], []], [[0]], [[0], []]
  ///
  /// \returns The newly created linalg::GenericOp consuming sharded operands.
  static linalg::GenericOp
  annotateGenericWithShard(linalg::GenericOp op, shard::GridOp grid,
                           const ShardPlan &plan,
                           llvm::StringMap<Value> &shardingCache) {

    // ------------------------------------------------------------------
    // Obs. on Shard dialect (based on ODS from LLVM commit: 064f02dac):
    //   %sh = shard.sharding @nsp split_axes = [...] : !shard.sharding
    //   %v2 = shard.shard %v to %sh annotate_for_users : tensor<...>
    //
    // The key point is: the "sharding description" is a Value
    // (!shard.sharding). Therefore, we must build shard.sharding first,
    // and then use it as the second operand to shard.shard.
    // ------------------------------------------------------------------

    OpBuilder b(op);
    Location loc = op.getLoc();
    MLIRContext *ctx = op.getContext();

    /// Build the `split_axes` specification for a single tensor operand.
    ///
    /// Given the operand's indexing map and ranked tensor type, this helper
    /// determines which tensor dimensions are sharded versus replicated.
    ///
    /// For each tensor dimension `d`, the dimension is marked as sharded if:
    ///   - The indexing map result for `d` is an AffineDimExpr, and
    ///   - That dimension expression refers exactly to `plan.shardIter`.
    ///
    /// In other words, a tensor dimension is sharded iff it is indexed
    /// directly by the loop iterator selected for sharding.
    ///
    /// This is operand-local by design. Therefore, different operands of the
    /// same generic may end up with different split_axes arrays, including
    /// different ranks for broadcast/projected operands.
    ///
    /// The result is an array of length equal to the tensor rank, where:
    ///   - An empty GridAxesAttr (`[]`) denotes replication.
    ///   - A GridAxesAttr containing `[0]` denotes sharding along grid axis 0.
    ///
    /// Even if all dimensions are replicated, an explicit rank-sized array
    /// is returned.
    auto buildSplitAxesForOperand =
        [&](AffineMap map,
            RankedTensorType rtt) -> SmallVector<shard::GridAxesAttr> {
      SmallVector<shard::GridAxesAttr> perDimAxes;
      perDimAxes.reserve(rtt.getRank());

      // map: (loops...) -> (tensor_dims...)
      for (int64_t d = 0; d < rtt.getRank(); ++d) {
        bool splitThisDim = false;

        // Only consider direct dim expressions for now.
        if (d < (int64_t)map.getNumResults()) {
          if (auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(d))) {
            if ((int64_t)dimExpr.getPosition() == plan.shardIter)
              splitThisDim = true;
          }
        }

        if (splitThisDim) {
          // Shard this tensor dimension on grid axis 0.
          perDimAxes.push_back(
              shard::GridAxesAttr::get(ctx, ArrayRef<int16_t>{0}));
        } else {
          // Replicated tensor dimension.
          perDimAxes.push_back(
              shard::GridAxesAttr::get(ctx, ArrayRef<int16_t>{}));
        }
      }

      // Note: Even if all entries are empty, we still return an array of size
      // 'rank' (e.g. `[[], [], ...]`), which is a valid "replicated" encoding.
      return perDimAxes;
    };

    /// Materialize a `!shard.sharding` SSA value for a tensor operand.
    ///
    /// This helper constructs an explicit sharding descriptor as a value,
    /// which can later be consumed by `shard.shard` ops and propagated through
    /// use-def chains.
    ///
    /// Behavior:
    ///  - If the operand is not a RankedTensorType, no sharding is created and
    ///    an empty Value is returned.
    ///  - Otherwise, the operand's indexing map is analyzed to compute
    ///    `split_axes` via `buildSplitAxesForOperand`.
    ///  - A `shard::ShardingOp` is created referencing the provided grid
    ///  symbol.
    ///
    /// Halo sizes and per-dimension offsets are intentionally left empty;
    /// they are not inferred at this stage of planning.
    ///
    /// \returns A Value of type `!shard.sharding`, or an empty Value if the
    ///          operand cannot be sharded.
    auto buildShardingValueFor = [&](AffineMap map, Value tensorVal) -> Value {
      auto rtt = dyn_cast<RankedTensorType>(tensorVal.getType());
      if (!rtt)
        return Value();

      // Reject maps we do not know how to interpret conservatively.
      //
      // This keeps the planner explicit for current bring-up and avoids
      // constructing misleading sharding descriptors for affine expressions
      // with arithmetic/symbols/repetitions.
      if (!isOperandMapCompatibleWithShardIter(map, plan.shardIter))
        return Value();

      // Build split_axes = [ GridAxesAttr, GridAxesAttr, ... ] (rank entries).
      SmallVector<shard::GridAxesAttr> splitAxes =
          buildSplitAxesForOperand(map, rtt);

      // Create the sharding descriptor value.
      return getOrCreateShardingValue(b, loc, grid, splitAxes, shardingCache);
    };

    // -------------------------------------------------------------------
    // Main body
    // -------------------------------------------------------------------

    // Indexing maps are ordered as: inputs..., outputs...
    auto maps = op.getIndexingMapsArray();

    // Wrap the generic op with the LinalgOp interface to access common Linalg
    // helpers.
    linalg::LinalgOp linalgOp(op);

    // Wrap each tensor input with shard.shard.
    SmallVector<Value> newInputs;
    newInputs.reserve(linalgOp.getNumDpsInputs());
    for (unsigned i = 0; i < linalgOp.getNumDpsInputs(); ++i) {
      Value in = op.getInputs()[i];
      AffineMap map = maps[i];
      Value shardingVal =
          getOrInferInputSharding(b, loc, in, map, buildShardingValueFor);
      if (!shardingVal) {
        newInputs.push_back(in);
        continue;
      }
      newInputs.push_back(getOrCreateShardFor(b, loc, ctx, in, shardingVal));
    }

    // Wrap each tensor output (init tensor) with shard.shard.
    SmallVector<Value> newOutputs;
    newOutputs.reserve(linalgOp.getNumDpsInits());
    for (unsigned oi = 0; oi < linalgOp.getNumDpsInits(); ++oi) {
      Value out = op.getOutputs()[oi];

      // The 'outs' operands of linalg.generic are *init tensors*
      // (destination-style). They must match the result type and be suitable
      // for SPMD partitioning.
      //
      // Some early pipelines produce linalg.generic with outs accidentally tied
      // to an input tensor (e.g. a read-modify-write pattern over the input).
      // This breaks shard-partition: the pass localizes the result type
      // (per-shard tile), but the outs operand remains global, triggering
      // verifier errors.
      //
      // Fix-up: if an outs operand aliases an input, synthesize a fresh init
      // tensor (tensor.empty) with the same type as the corresponding op
      // result.
      auto aliasesAnyInput = [&]() -> bool {
        for (Value in : op.getInputs())
          if (in == out)
            return true;
        return false;
      };

      if (aliasesAnyInput() && oi < op->getNumResults()) {
        auto resTy = dyn_cast<RankedTensorType>(op->getResult(oi).getType());
        if (resTy) {
          SmallVector<Value> dynSizes;
          dynSizes.reserve(resTy.getNumDynamicDims());

          // For dynamic dimensions, use tensor.dim on the first input as a size
          // source. This is safe for elementwise ops where shapes are expected
          // to match.
          Value sizeSource =
              !op.getInputs().empty() ? op.getInputs().front() : Value();
          for (int64_t d = 0, e = resTy.getRank(); d < e; ++d) {
            if (resTy.isDynamicDim(d)) {
              // If we don't have a size source, we can't build a well-formed
              // empty. Keep the original outs in that (rare) case.
              if (!sizeSource)
                break;
              dynSizes.push_back(b.create<tensor::DimOp>(loc, sizeSource, d));
            }
          }
          if ((int64_t)dynSizes.size() == resTy.getNumDynamicDims()) {
            out = b.create<tensor::EmptyOp>(loc, resTy.getShape(),
                                            resTy.getElementType(), dynSizes);
          }
        }
      }

      AffineMap map = maps[linalgOp.getNumDpsInputs() + oi];
      Value shardingVal = buildShardingValueFor(map, out);
      if (!shardingVal) {
        newOutputs.push_back(out);
        continue;
      }
      auto shardOp =
          b.create<shard::ShardOp>(loc, out.getType(),
                                   /*src=*/out,
                                   /*sharding=*/shardingVal,
                                   /*annotate_for_users=*/UnitAttr::get(ctx));
      newOutputs.push_back(shardOp.getResult());
    }

    // Rebuild the linalg::GenericOp so that it consumes the sharded SSA values.
    //
    // Since sharding is represented as SSA data, simply inserting shard.shard
    // ops is insufficient. To ensure the computation consumes them, we must
    // create a new linalg.generic with (newInputs/newOutputs), then move the
    // region and replace the old op.
    b.setInsertionPoint(op);

    SmallVector<AffineMap> indexingMaps(op.getIndexingMapsArray());
    SmallVector<utils::IteratorType> iteratorTypes(op.getIteratorTypesArray());

    auto newOp = b.create<linalg::GenericOp>(loc, op->getResultTypes(),
                                             /*inputs=*/newInputs,
                                             /*outputs=*/newOutputs,
                                             /*indexingMaps=*/indexingMaps,
                                             /*iteratorTypes=*/iteratorTypes);

    // Preserve any extra attributes on the original generic op.
    // Avoid overwriting the structural attributes that define the generic
    // itself.
    auto indexingMapsName = StringAttr::get(op->getContext(), "indexing_maps");
    auto iteratorTypesName =
        StringAttr::get(op->getContext(), "iterator_types");
    for (auto na : op->getAttrs()) {
      auto name = na.getName();
      if (name == indexingMapsName || name == iteratorTypesName)
        continue;
      newOp->setAttr(name, na.getValue());
    }

    // Move the region body into the new generic.
    newOp.getRegion().takeBody(op.getRegion());

    // Replace all uses and erase the old op.
    op->replaceAllUsesWith(newOp->getResults());
    op->erase();

    return newOp;
  }

  /// Insert an all-reduce if the sharding plan requires it.
  ///
  /// This is relevant for global reductions (e.g. sum-reduction) where each
  /// NSP produces a partial result and the program expects a replicated final
  /// value.
  static void materializeAllReduceIfNeeded(linalg::GenericOp op,
                                           const ShardPlan &plan) {
    // Skeleton:
    //  * Identify the value that represents the partial reduction.
    //  * Insert shard.all_reduce with op="add" (or other reduction kind).
    //
    // This can be done by a separate "collective materialization" pass;
    // keeping it explicit here helps debugging and documentation.
    (void)op;
    (void)plan;
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass registration.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace hexagon {

std::unique_ptr<Pass> createNSPShardPlannerPass() {
  return std::make_unique<NSPShardPlannerPass>();
}

std::unique_ptr<Pass> createNSPShardPlannerPass(int64_t nspCount,
                                                bool allowCollectives) {
  ShardPolicy policy;
  policy.nspCount = nspCount;
  policy.allowCollectives = allowCollectives;
  return std::make_unique<NSPShardPlannerPass>(policy);
}

} // namespace hexagon
} // namespace mlir
