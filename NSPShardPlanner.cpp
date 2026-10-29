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
// annotations based on iterator_types and indexing_maps. It also records when
// a sharding decision would require cross-NSP collectives (e.g. all-reduce),
// but collective materialization is not currently implemented here.
//
// Design goals:
//   * Work at the "linalg.generic" level (pre-tiling).
//   * Be dialect-agnostic except for using the shard dialect as the sharding
//   IR.
//   * Keep the pipeline compatible with further lowering to per-NSP slices
//     (tensor.extract_slice/memref.subview) and runtime intrinsics.
//
// Example:
//   Given an elementwise tensor computation:
//   ```mlir
//   %0 = linalg.generic ... ins(%arg0, %arg1 : tensor<2048x512xf32>,
//                                            tensor<2048x512xf32>)
//                         outs(%init : tensor<2048x512xf32>) ->
//                         tensor<2048x512xf32>
//   ```
//   the planner does not tile or rewrite the computation yet.  Instead, for a
//   1-D NSP grid with 32 participants, it records that dimension 0 is split:
//   ```mlir
//   shard.grid @nsp(shape = 32)
//   %s = shard.sharding @nsp split_axes = [[0], []]
//   %a = shard.shard %arg0 to %s annotate_for_users
//   %b = shard.shard %arg1 to %s annotate_for_users
//   %1 = linalg.generic ... ins(%a, %b : tensor<2048x512xf32>,
//                                      tensor<2048x512xf32>)
//                         outs(%init : tensor<2048x512xf32>) ->
//                         tensor<2048x512xf32>
//   ```
//   Later passes consume these annotations and turn the global tensor operation
//   into per-NSP local work, for example slices of shape tensor<64x512xf32>.
//   Thus this pass is only the planner: it describes the intended partitioning
//   without materializing the local slices.
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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
#include <utility>

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
struct ShardPolicy {
  // Requested logical grid shape.  The current planning policy still uses the
  // flattened product of these axes for one tensor dimension, but keeping cores
  // and threads separate in shard.grid lets later planning distinguish them.
  int64_t numCores = 16;
  int64_t numThreads = 4;

  int64_t getNumShards() const { return numCores * numThreads; }

  // Prefer sharding along the first parallel iterator that indexes the output.
  bool preferFirstOutputParallelDim = true;

  // Currently sharding reduction is not supported.
  bool allowCollectives = false;
};

/// Normalized information read from the actual shard.grid symbol.
struct GridShapeInfo {
  int64_t numShards = 0;
  SmallVector<int64_t, 2> axisExtents;
  SmallVector<int16_t, 2> flattenedGridAxes;

  bool is2DGrid() const { return axisExtents.size() == 2; }

  int64_t getAxisExtent(int16_t axis) const {
    if (axis < 0 || axis >= static_cast<int16_t>(axisExtents.size()))
      return 0;
    return axisExtents[axis];
  }
};

/// Map one linalg loop iterator to one or more grid axes.
struct LoopGridAxisMapping {
  int64_t loopIter = -1;
  SmallVector<int16_t, 2> gridAxes;
};

/// A structured description of the sharding decision for a single op.
struct ShardPlan {
  // Primary loop iterator.  The legacy path maps only this iterator to the
  // flattened grid.  Multi-axis plans may map additional iterators below.
  int64_t shardIter = -1;

  // Whether this plan requires a cross-NSP all-reduce to be correct.
  bool requiresAllReduce = false;

  // Per-loop grid-axis mapping.  Legacy elementwise plans contain one entry
  // mapping `shardIter` to all flattened grid axes.  Matmul plans may contain
  // multiple entries, e.g. i -> axis 0 and j -> axis 1.
  SmallVector<LoopGridAxisMapping, 2> loopGridAxes;

  void addLoopGridAxes(int64_t loopIter, ArrayRef<int16_t> axes) {
    LoopGridAxisMapping mapping;
    mapping.loopIter = loopIter;
    mapping.gridAxes.assign(axes.begin(), axes.end());
    loopGridAxes.push_back(std::move(mapping));
  }

  ArrayRef<int16_t> getGridAxesForLoopIter(int64_t loopIter) const {
    for (const LoopGridAxisMapping &mapping : loopGridAxes)
      if (mapping.loopIter == loopIter)
        return mapping.gridAxes;
    return {};
  }
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
///  - The default implementation may use a 2-D shard grid, but flattens all
///    grid axes onto the selected tensor dimension to preserve legacy behavior.
///  - Matmul-like contractions may map output parallel iterators to distinct
///    grid axes, allowing cores and threads to model independent tensor axes.
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
    shard::GridOp grid =
        getOrCreateGrid(module, policy.numCores, policy.numThreads);
    FailureOr<GridShapeInfo> gridInfo = getGridShapeInfo(grid);
    if (failed(gridInfo)) {
      signalPassFailure();
      return;
    }

    llvm::StringMap<Value> shardingCache;

    // We rebuild (and erase) linalg.generic ops during annotation.
    // Erasing ops while walking can invalidate the walk. We collect
    // first, then rewrite in a stable loop.
    SmallVector<linalg::GenericOp> generics;
    SmallVector<linalg::GenericOp> scatterGenerics;

    func.walk([&](linalg::GenericOp op) {
      linalg::LinalgOp linalgOp(op);
      // Scatter-only generics (ins-only, no results, body has memref.store)
      // must be checked before hasPureTensorSemantics because they satisfy
      // the "no memref operands + at least one tensor" condition but have
      // zero DPS inits and would fail buildPlanForGeneric.
      if (isScatterOnlyGeneric(op))
        scatterGenerics.push_back(op);
      else if (linalgOp.hasPureTensorSemantics())
        generics.push_back(op);
    });

    for (linalg::GenericOp op : generics) {
      // The op may have been erased by a previous rewrite.
      if (!op || !op->getParentOp())
        continue;

      // Skip ops whose inputs are Triton-style pre-loaded program_id tiles.
      // Sharding such ops only covers one tile of the full iteration space.
      if (hasPreloadedTileInput(op))
        continue;

      // Analyze the op and build a sharding plan.
      FailureOr<ShardPlan> plan = buildPlanForGeneric(op, *gridInfo);
      if (failed(plan))
        continue;

      // If the plan introduces collectives but they are not allowed, skip
      // this op in the current no-communication mode.
      if (plan->requiresAllReduce && !policy.allowCollectives)
        continue;

      // Attach shard annotations to operands using shard.shard AND rebuild
      // the linalg.generic so that it consumes the annotated values.
      linalg::GenericOp newOp =
          annotateGenericWithShard(op, grid, *plan, shardingCache);

      // Collective materialization is not currently implemented in the
      // NSPSharding pipeline. Keep requiresAllReduce as analysis information in
      // the plan, but fail explicitly instead of silently pretending support
      // exists.
      //
      // At the moment, NSPLocalize/NSPMaterialize/ShardToLLVM only support the
      // no-communication path. Global reductions that require cross-NSP
      // aggregation need a future dedicated collective materialization flow.
      if (plan->requiresAllReduce) {
        newOp.emitError()
            << "NSP collective materialization is not currently implemented; "
               "this sharding plan requires an all-reduce for correctness";
        signalPassFailure();
        return;
      }
    }

    // Annotate scatter-only (ins-only, no results) generics.
    for (linalg::GenericOp op : scatterGenerics) {
      if (!op || !op->getParentOp())
        continue;

      // Skip scatter generics whose input tensors were computed by a
      // linalg generic using linalg.index (program_id-based offset).  With
      // NUM_ITERATIONS_SEMANTICS the program_id is non-zero on all NSPs except
      // NSP-0, so every NSP would compute a different shifted window of the
      // global input buffer.  Sharding the scatter while leaving the parent
      // computation unsharded causes out-of-bounds reads on those NSPs.
      if (scatterInputComputedFromLinalgIndex(op))
        continue;

      FailureOr<ShardPlan> plan = buildPlanForScatterGeneric(op, *gridInfo);
      if (failed(plan))
        continue;

      annotateScatterGenericWithShard(op, grid, *plan, shardingCache);
    }
  }

private:
  ShardPolicy policy;

  /// Get or create `shard.grid @nsp(shape = [numCores, numThreads])`.
  static shard::GridOp getOrCreateGrid(ModuleOp module, int64_t numCores,
                                       int64_t numThreads) {
    MLIRContext *ctx = module.getContext();
    OpBuilder b(ctx);

    // Look up a pre-existing grid named "nsp".
    if (auto grid = module.lookupSymbol<shard::GridOp>("nsp"))
      return grid;

    b.setInsertionPointToStart(module.getBody());
    SmallVector<int64_t, 2> shape = {numCores, numThreads};
    return shard::GridOp::create(b, module.getLoc(), "nsp",
                                 llvm::ArrayRef<int64_t>(shape));
  }

  /// Normalize both legacy 1-D grids and the new 2-D NSP grid.
  static FailureOr<GridShapeInfo> getGridShapeInfo(shard::GridOp grid) {
    ArrayRef<int64_t> shape = grid.getShape();
    if (shape.empty() || shape.size() > 2) {
      grid.emitOpError() << "expects a 1-D or 2-D NSP grid shape";
      return failure();
    }

    GridShapeInfo info;
    info.numShards = 1;
    info.axisExtents.reserve(shape.size());
    info.flattenedGridAxes.reserve(shape.size());

    for (auto [axis, extent] : llvm::enumerate(shape)) {
      if (extent <= 0) {
        grid.emitOpError() << "has invalid non-positive grid extent " << extent
                           << " at axis " << axis;
        return failure();
      }
      info.numShards *= extent;
      info.axisExtents.push_back(extent);
      info.flattenedGridAxes.push_back(static_cast<int16_t>(axis));
    }

    return info;
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
  ///
  /// The descriptor is hoisted to the function entry block. Hoisting
  /// descriptors keeps the per-function cache safe for both nested and
  /// outer users.
  static Value getOrCreateShardingValue(func::FuncOp func, Location loc,
                                        shard::GridOp grid,
                                        ArrayRef<shard::GridAxesAttr> splitAxes,
                                        llvm::StringMap<Value> &shardingCache) {
    if (!func || func.getBody().empty())
      return Value();

    std::string key = buildShardingCacheKey(grid, splitAxes);

    if (auto it = shardingCache.find(key); it != shardingCache.end())
      return it->second;

    OpBuilder entryBuilder(func.getContext());
    entryBuilder.setInsertionPointToStart(&func.getBody().front());

    auto gridRef = FlatSymbolRefAttr::get(grid.getSymNameAttr());

    auto shardingOp = shard::ShardingOp::create(
        entryBuilder, loc,
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
  static bool isOperandMapCompatibleWithShardPlan(AffineMap map,
                                                  const ShardPlan &plan) {
    (void)plan;

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

  /// Create a new shard annotation op for (`src`, `sharding`).
  ///
  /// Existing shard.shard users of `src` may have been created inside a
  /// nested region and may not dominate the current insertion point.
  static Value getOrCreateShardFor(OpBuilder &b, Location loc, MLIRContext *ctx,
                                   Value src, Value sharding) {

    auto shardOp =
        shard::ShardOp::create(b, loc, src.getType(),
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

  /// Return the tensor result dimension indexed directly by `loopIter`.
  static std::optional<int64_t> getTensorDimForLoopIter(AffineMap map,
                                                        int64_t loopIter) {
    for (auto [tensorDim, expr] : llvm::enumerate(map.getResults())) {
      auto dim = dyn_cast<AffineDimExpr>(expr);
      if (dim && static_cast<int64_t>(dim.getPosition()) == loopIter)
        return static_cast<int64_t>(tensorDim);
    }
    return std::nullopt;
  }

  /// Build a single-axis matmul plan using the full NSP grid.
  ///
  /// Prefer the larger output dimension (M on ties), but reject dimensions
  /// that cannot be split evenly or would produce tiles smaller than 16.
  /// Leave the reduction dimension K unsharded.
  static FailureOr<ShardPlan> buildMatmul1DPlan(
      linalg::GenericOp op, const GridShapeInfo &gridInfo,
      RankedTensorType outputTy, int64_t iIter, int64_t jIter) {
    constexpr int64_t minTileExtent = 16;
    if (!outputTy || !outputTy.hasStaticShape() || gridInfo.numShards <= 0)
      return failure();

    linalg::LinalgOp linalgOp(op);
    AffineMap outMap = op.getIndexingMapsArray()[linalgOp.getNumDpsInputs()];
    std::optional<int64_t> iDim = getTensorDimForLoopIter(outMap, iIter);
    std::optional<int64_t> jDim = getTensorDimForLoopIter(outMap, jIter);
    if (!iDim || !jDim || *iDim == *jDim)
      return failure();

    int64_t m = outputTy.getDimSize(*iDim);
    int64_t n = outputTy.getDimSize(*jDim);
    auto canShard = [&](int64_t extent) {
      return extent > 0 && extent % gridInfo.numShards == 0 &&
             extent / gridInfo.numShards >= minTileExtent;
    };

    int64_t preferredIter = m >= n ? iIter : jIter;
    int64_t alternateIter = m >= n ? jIter : iIter;
    int64_t preferredExtent = m >= n ? m : n;
    int64_t alternateExtent = m >= n ? n : m;

    int64_t chosenIter = -1;
    if (canShard(preferredExtent))
      chosenIter = preferredIter;
    else if (canShard(alternateExtent))
      chosenIter = alternateIter;
    else
      return failure();

    ShardPlan plan;
    plan.shardIter = chosenIter;
    plan.requiresAllReduce = false;
    plan.addLoopGridAxes(chosenIter, gridInfo.flattenedGridAxes);
    return plan;
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
  static FailureOr<ShardPlan>
  buildPlanForGeneric(linalg::GenericOp op, const GridShapeInfo &gridInfo) {
    // Extract iterator types.
    SmallVector<mlir::utils::IteratorType> iters;
    iters.reserve(op.getNumLoops());
    for (mlir::utils::IteratorType it : op.getIteratorTypesArray())
      iters.push_back(it);

    // Identify output indexing map.
    // We assume single output for simplicity (generalize as needed).
    linalg::LinalgOp linalgOp(op);
    if (linalgOp.getNumDpsInits() != 1)
      return failure();
    AffineMap outMap = op.getIndexingMapsArray().back();
    auto outTy = dyn_cast<RankedTensorType>(op.getResult(0).getType());

    // Matmul-like generics must use exactly one output dimension for now.
    // Do not fall back to generic planning when neither M nor N is viable.
    int64_t iIter = -1, jIter = -1, kIter = -1;
    if (matchMatmulLike(op, iIter, jIter, kIter)) {
      FailureOr<ShardPlan> matmulPlan =
          buildMatmul1DPlan(op, gridInfo, outTy, iIter, jIter);
      if (failed(matmulPlan))
        return failure();
      for (AffineMap map : op.getIndexingMapsArray())
        if (!isOperandMapCompatibleWithShardPlan(map, *matmulPlan))
          return failure();
      return matmulPlan;
    }

    ShardPlan plan;

    // Pick a sharding iterator for the legacy flattened path.
    if (auto transposeChoice = pickShardIteratorFromTransposeInput(op))
      plan.shardIter = *transposeChoice;
    else
      plan.shardIter =
          pickShardIteratorIndex(outMap, iters, outTy, gridInfo.numShards);
    if (plan.shardIter < 0)
      return failure();

    plan.addLoopGridAxes(plan.shardIter, gridInfo.flattenedGridAxes);

    // Validate that every operand/result map is compatible with deriving
    // operand-local split_axes from the chosen shard iterator(s).
    //
    // This intentionally allows heterogenous operand shardings, including
    // different ranks, as long as they are all consistent with the same
    // loop-iterator decision. This is required for broadcast-style patterns
    // such as:
    //   ins(tensor<MxN>, tensor<M>) -> outs(tensor<MxN>)
    for (AffineMap map : op.getIndexingMapsArray()) {
      if (!isOperandMapCompatibleWithShardPlan(map, plan))
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

  /// Decide which iterator to shard based on output map, iterator_types and,
  /// when available, the static output shape.
  /// \returns the loop iterator index, or -1 if no suitable iterator exists.
  static int64_t
  pickShardIteratorIndex(AffineMap outMap,
                         ArrayRef<mlir::utils::IteratorType> iteratorTypes,
                         RankedTensorType outputTy, int64_t numShards) {

    struct Candidate {
      int64_t iterIdx;
      int64_t resultDim;
    };

    SmallVector<Candidate> candidates;

    // We want an iterator that:
    //   (1) is "parallel"
    //   (2) appears as a plain AffineDimExpr in the output map results.
    //
    // Example: outMap = (i,j,k) -> (i,j)
    // candidates: i and j (if both are parallel iterators).
    for (auto [resultDim, expr] : llvm::enumerate(outMap.getResults())) {
      auto dim = dyn_cast<AffineDimExpr>(expr);
      if (!dim)
        continue;
      int64_t iterIdx = dim.getPosition();
      if (iterIdx < 0 || iterIdx >= (int64_t)iteratorTypes.size())
        continue;
      if (iteratorTypes[iterIdx] != mlir::utils::IteratorType::parallel)
        continue;
      candidates.push_back(Candidate{iterIdx, static_cast<int64_t>(resultDim)});
    }

    if (candidates.empty())
      return -1;

    // Choose a dimension that can be evenly split by the NSP grid. This avoids
    // selecting a leading unit/batch dimension where axis 0 cannot be
    // partitioned.
    if (outputTy && outputTy.hasStaticShape()) {
      for (Candidate candidate : candidates) {
        if (candidate.resultDim < 0 ||
            candidate.resultDim >= outputTy.getRank())
          continue;
        int64_t extent = outputTy.getDimSize(candidate.resultDim);
        if (extent > 0 && numShards > 0 && extent % numShards == 0)
          return candidate.iterIdx;
      }
    }

    // Fallback to the previous behavior: first parallel iterator in output
    // order. This preserves the old planning behavior for dynamic shapes or
    // partially unsupported cases.
    return candidates.front().iterIdx;
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
  /// iterator(s) to shard have already been chosen).
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
    func::FuncOp func = op->getParentOfType<func::FuncOp>();

    /// Build the `split_axes` specification for a single tensor operand.
    ///
    /// Given the operand's indexing map and ranked tensor type, this helper
    /// determines which tensor dimensions are sharded versus replicated.
    ///
    /// For each tensor dimension `d`, the dimension is marked as sharded if:
    ///   - The indexing map result for `d` is an AffineDimExpr, and
    ///   - That dimension expression refers to a loop iterator mapped by the
    ///     plan to one or more grid axes.
    ///
    /// In other words, a tensor dimension is sharded iff it is indexed
    /// directly by a loop iterator selected for sharding.
    ///
    /// This is operand-local by design. Therefore, different operands of the
    /// same generic may end up with different split_axes arrays, including
    /// different ranks for broadcast/projected operands.
    ///
    /// The result is an array of length equal to the tensor rank, where:
    ///   - An empty GridAxesAttr (`[]`) denotes replication.
    ///   - A GridAxesAttr containing the mapped plan grid axes denotes sharding
    ///     along those NSP grid axes.
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
        ArrayRef<int16_t> mappedGridAxes;

        // Only consider direct dim expressions for now.
        if (d < (int64_t)map.getNumResults()) {
          if (auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(d))) {
            mappedGridAxes = plan.getGridAxesForLoopIter(dimExpr.getPosition());
          }
        }

        if (!mappedGridAxes.empty()) {
          // Shard this tensor dimension on the mapped NSP grid axes.
          perDimAxes.push_back(shard::GridAxesAttr::get(ctx, mappedGridAxes));
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
      if (!isOperandMapCompatibleWithShardPlan(map, plan))
        return Value();

      // Build split_axes = [ GridAxesAttr, GridAxesAttr, ... ] (rank entries).
      SmallVector<shard::GridAxesAttr> splitAxes =
          buildSplitAxesForOperand(map, rtt);

      // Create the sharding descriptor value in a dominance-safe location.
      return getOrCreateShardingValue(func, loc, grid, splitAxes,
                                      shardingCache);
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

      // The 'outs' operands of linalg.generic are init tensors. Reusing the
      // same init tensor across multiple producers is problematic for shard
      // propagation when those producers imply different logical shardings
      // (e.g. transpose result vs non-transposed elementwise result).
      //
      // Therefore, besides the existing input-aliasing fix-up, also materialize
      // a fresh init tensor when the outs value is shared.

      auto aliasesAnyInput = [&]() -> bool {
        for (Value in : op.getInputs())
          if (in == out)
            return true;
        return false;
      };

      auto hasMultipleUses = [&](Value v) -> bool { return !v.hasOneUse(); };

      auto needsFreshInit = [&]() -> bool {
        if (aliasesAnyInput())
          return true;
        if (hasMultipleUses(out))
          return true;
        return false;
      };

      if (needsFreshInit() && oi < op->getNumResults()) {
        auto resTy = dyn_cast<RankedTensorType>(op->getResult(oi).getType());
        if (resTy) {
          SmallVector<Value> dynSizes;
          dynSizes.reserve(resTy.getNumDynamicDims());

          // For dynamic dimensions, use tensor.dim on the first input as a size
          // source. This is safe for elementwise/projected cases where shapes
          // are expected to match the result shape.
          Value sizeSource =
              !op.getInputs().empty() ? op.getInputs().front() : Value();
          for (int64_t d = 0, e = resTy.getRank(); d < e; ++d) {
            if (resTy.isDynamicDim(d)) {
              if (!sizeSource)
                break;
              dynSizes.push_back(tensor::DimOp::create(b, loc, sizeSource, d));
            }
          }
          if ((int64_t)dynSizes.size() == resTy.getNumDynamicDims()) {
            out = tensor::EmptyOp::create(b, loc, resTy.getShape(),
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
          shard::ShardOp::create(b, loc, out.getType(),
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

    auto newOp = linalg::GenericOp::create(b, loc, op->getResultTypes(),
                                           /*inputs=*/newInputs,
                                           /*outputs=*/newOutputs,
                                           /*indexingMaps=*/indexingMaps,
                                           /*iteratorTypes=*/iteratorTypes);

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

  /// Return true iff `op` is a scatter-only generic:
  ///   - At least one ranked tensor input.
  ///   - Zero DPS inits and zero tensor results.
  ///   - The body contains at least one memref.store op.
  /// These are produced by Triton kernels with non-linear scatter stores.
  static bool isScatterOnlyGeneric(linalg::GenericOp op) {
    linalg::LinalgOp linalgOp(op);
    if (linalgOp.getNumDpsInputs() < 1)
      return false;
    if (linalgOp.getNumDpsInits() != 0)
      return false;
    if (op->getNumResults() != 0)
      return false;
    // Ensure at least one input is a ranked tensor.
    bool hasRankedTensorInput = false;
    for (Value in : op.getInputs()) {
      if (dyn_cast<RankedTensorType>(in.getType())) {
        hasRankedTensorInput = true;
        break;
      }
    }
    if (!hasRankedTensorInput)
      return false;
    // Body must contain at least one memref.store.
    bool hasStore = false;
    op.getRegion().front().walk(
        [&](mlir::memref::StoreOp) { hasStore = true; });
    return hasStore;
  }

  /// Return true if the reinterpret_cast has a dynamic (non-zero-constant)
  /// offset - meaning it is indexed by a runtime value such as
  /// program_id*BLOCK. A static zero offset covers the whole tensor from the
  /// start and is safe to shard; a dynamic offset means the copy only covers
  /// one program_id tile.
  static bool reinterpretCastHasDynamicOffset(memref::ReinterpretCastOp rc) {
    auto offsets = rc.getMixedOffsets();
    if (offsets.empty())
      return false;
    if (auto constVal = getConstantIntValue(offsets[0]))
      return *constVal != 0;
    return true;
  }

  /// Return true if `op` has a to_tensor(alloc) input whose alloc is filled by
  /// a memref.copy from a reinterpret_cast(memref<*xT>) with a DYNAMIC offset
  /// (i.e. program_id-indexed tile).
  ///
  /// Static-zero-offset casts cover the entire tensor and are safe to shard.
  ///
  /// Use this for the scatter path (no scf.for exclusion needed: even a gather
  /// loop over a dynamic-offset cast is program_id-tied).
  static bool hasAnyAllocCopyInput(linalg::GenericOp op) {
    for (Value operand : op->getOperands()) {
      auto toTensor = operand.getDefiningOp<bufferization::ToTensorOp>();
      if (!toTensor)
        continue;
      auto alloc = toTensor->getOperand(0).getDefiningOp<memref::AllocOp>();
      if (!alloc)
        continue;
      for (Operation *user : alloc.getResult().getUsers()) {
        auto subview = dyn_cast<memref::SubViewOp>(user);
        if (!subview)
          continue;
        for (Operation *svUser : subview.getResult().getUsers()) {
          auto copy = dyn_cast<memref::CopyOp>(svUser);
          if (!copy || copy.getTarget() != subview.getResult())
            continue;
          Value src = copy.getSource();
          if (auto srcSv = src.getDefiningOp<memref::SubViewOp>())
            src = srcSv.getSource();
          if (auto rc = src.getDefiningOp<memref::ReinterpretCastOp>()) {
            if (!dyn_cast<BaseMemRefType>(rc.getSource().getType()).hasRank() &&
                reinterpretCastHasDynamicOffset(rc))
              return true;
          }
        }
      }
    }
    return false;
  }

  /// Return true if any tensor input of `op` is a Triton-style pre-loaded
  /// contiguous tile from a program_id-indexed dynamic-offset reinterpret_cast,
  /// copied outside a scf.for (bulk preload, not gather-style).
  ///
  /// Two patterns are recognized:
  ///
  /// Pattern A (indirect, via alloc+copy):
  ///   to_tensor(alloc) where alloc is filled by
  ///   memref.copy(subview(alloc), reinterpret_cast<*xT>[program_id*BLOCK])
  ///   outside a scf.for.
  ///
  /// Pattern B (direct):
  ///   to_tensor(reinterpret_cast<*xT>[program_id*BLOCK])
  ///   The inductor silu/unsafe_view_mul kernel uses this form: the
  ///   reinterpret_cast is passed directly to to_tensor without an
  ///   alloc+copy intermediary. Without detecting this pattern the planner
  ///   annotates the generic for sharding, and the NSPMaterializePass later
  ///   computes an out-of-bounds store offset (Error 500).
  ///
  /// Static-zero-offset casts and gather loops (scf.for) are both safe to
  /// shard.
  ///
  /// Use this for the regular (output-bearing) generic path only.
  static bool hasPreloadedTileInput(linalg::GenericOp op) {
    for (Value operand : op->getOperands()) {
      auto toTensor = operand.getDefiningOp<bufferization::ToTensorOp>();
      if (!toTensor) {
        // Pattern C: operand is the result of a scf.for whose body loads
        // program_id-indexed tiles via to_tensor(reinterpret_cast[dynamic]).
        // The for-loop output is a locally-computed tile (e.g. the flash
        // attention accumulator `acc`). Sharding it would incorrectly divide
        // the local result instead of the global tensor.
        if (auto forResult = dyn_cast<OpResult>(operand))
          if (auto forOp = dyn_cast<scf::ForOp>(forResult.getOwner())) {
            bool found = false;
            forOp.walk([&](bufferization::ToTensorOp tt) {
              Value src = tt->getOperand(0);
              if (auto rc = src.getDefiningOp<memref::ReinterpretCastOp>())
                if (!dyn_cast<BaseMemRefType>(rc.getSource().getType())
                         .hasRank() &&
                    reinterpretCastHasDynamicOffset(rc))
                  found = true;
            });
            if (found)
              return true;
          }
        continue;
      }

      Value toTensorSrc = toTensor->getOperand(0);

      // Pattern B: direct to_tensor(reinterpret_cast) with dynamic offset.
      if (auto rc = toTensorSrc.getDefiningOp<memref::ReinterpretCastOp>()) {
        if (!dyn_cast<BaseMemRefType>(rc.getSource().getType()).hasRank() &&
            reinterpretCastHasDynamicOffset(rc))
          return true;
      }

      // Pattern A: to_tensor(alloc) where alloc is filled via copy from a
      // dynamic-offset reinterpret_cast outside a scf.for.
      auto alloc = toTensorSrc.getDefiningOp<memref::AllocOp>();
      if (!alloc)
        continue;
      for (Operation *user : alloc.getResult().getUsers()) {
        auto subview = dyn_cast<memref::SubViewOp>(user);
        if (!subview)
          continue;
        for (Operation *svUser : subview.getResult().getUsers()) {
          auto copy = dyn_cast<memref::CopyOp>(svUser);
          if (!copy || copy.getTarget() != subview.getResult())
            continue;
          // Copies inside a scf.for are gather-style - not a bulk preload.
          if (copy->getParentOfType<scf::ForOp>())
            continue;
          Value src = copy.getSource();
          if (auto srcSv = src.getDefiningOp<memref::SubViewOp>())
            src = srcSv.getSource();
          if (auto rc = src.getDefiningOp<memref::ReinterpretCastOp>()) {
            if (!dyn_cast<BaseMemRefType>(rc.getSource().getType().hasRank() &&
                reinterpretCastHasDynamicOffset(rc))
              return true;
          }
        }
      }
    }
    return false;
  }

  /// Return true if any ranked-tensor input of the scatter generic is
  /// produced by a linalg.generic whose body contains a linalg.index op.
  ///
  /// Such inputs are computed with an offset of `program_id * BLOCK` baked in.
  /// With NUM_ITERATIONS_SEMANTICS the program_id is non-zero on NSPs other
  /// than 0, so every NSP would read a different (shifted) window of the
  /// global buffer. Sharding the scatter while leaving the parent computation
  /// unsharded causes out-of-bounds accesses on those NSPs.
  static bool scatterInputComputedFromLinalgIndex(linalg::GenericOp op) {
    for (Value input : op.getInputs()) {
      auto *defOp = input.getDefiningOp();
      if (!defOp)
        continue;
      // Walk through insert_slice / extract_result wrappers to the real
      // source.
      if (auto res = dyn_cast<OpResult>(input)) {
        auto parentGeneric = dyn_cast<linalg::GenericOp>(res.getOwner());
        if (parentGeneric) {
          bool hasLinalgIndex = false;
          parentGeneric.getRegion().front().walk(
              [&](linalg::IndexOp) { hasLinalgIndex = true; });
          if (hasLinalgIndex)
            return true;
        }
      }
    }
    return false;
  }

  /// Build a sharding plan for a scatter-only generic.
  ///
  /// Since the op has no output tensor we derive the shard iterator from the
  /// first input's indexing map instead of an output map.
  static FailureOr<ShardPlan>
  buildPlanForScatterGeneric(linalg::GenericOp op,
                             const GridShapeInfo &gridInfo) {
    SmallVector<mlir::utils::IteratorType> iters;
    iters.reserve(op.getNumLoops());
    for (mlir::utils::IteratorType it : op.getIteratorTypesArray())
      iters.push_back(it);

    auto maps = op.getIndexingMapsArray();
    if (maps.empty())
      return failure();

    // Use the first input map as the "output proxy" for iterator selection.
    AffineMap inMap = maps[0];
    auto inTy = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());

    ShardPlan plan;
    plan.shardIter =
        pickShardIteratorIndex(inMap, iters, inTy, gridInfo.numShards);
    if (plan.shardIter < 0) {
      // Warn when the tensor is too small to divide across all NSPs.  The most
      // common cause is BLOCK_SIZE < nspCount (e.g. BLOCK_SIZE=16, 32 NSPs).
      if (inTy && inTy.hasStaticShape() && gridInfo.numShards > 0) {
        int64_t maxDim = *llvm::max_element(inTy.getShape());
        if (maxDim < gridInfo.numShards)
          op.emitWarning()
              << "NSPShardPlanner: scatter generic input has max dim " << maxDim
              << " < nspCount " << gridInfo.numShards
              << "; no sharding applied - increase BLOCK_SIZE to >= "
              << gridInfo.numShards;
      }
      return failure();
    }

    plan.addLoopGridAxes(plan.shardIter, gridInfo.flattenedGridAxes);
    plan.requiresAllReduce = false;

    // Check that every input map is compatible.
    for (AffineMap map : maps) {
      if (!isOperandMapCompatibleWithShardPlan(map, plan))
        return failure();
    }

    return plan;
  }

  /// Annotate a scatter-only (ins-only, no results) linalg.generic with shard
  /// planning information encoded as op attributes instead of shard.shard SSA
  /// values.
  ///
  /// Rationale: scatter generics live in a mixed tensor+memref context.
  /// Wrapping their inputs in shard.shard would trigger sharding-propagation,
  /// which walks the entire function and fails on ops like memref.cast that
  /// do not implement the sharding interface. Using attributes avoids that
  /// issue while still giving NSPLocalizePass everything it needs.
  ///
  /// Emitted attributes on the linalg.generic:
  ///   nsp.scatter_shard      : unit   (marker for NSPLocalizePass)
  ///   nsp.scatter_num_shards : i64    (flattened grid size)
  ///   nsp.scatter_grid       : string (grid symbol name, e.g. "@nsp")
  static void annotateScatterGenericWithShard(linalg::GenericOp op,
                                              shard::GridOp grid,
                                              const ShardPlan &plan,
                                              llvm::StringMap<Value> &) {
    MLIRContext *ctx = op.getContext();

    // Compute the total shard count from the plan's grid axes.
    // For a 1-D or flattened plan the shardIter maps to all grid axes; we
    // read the numShards from the plan's flattenedGridAxes product.
    // However plan does not store extents; we derive numShards from the
    // grid shape.
    int64_t numShards = 1;
    ArrayRef<int64_t> shape = grid.getShape();
    for (int64_t ext : shape)
      numShards *= ext;

    op->setAttr("nsp.scatter_shard", UnitAttr::get(ctx));
    op->setAttr("nsp.scatter_num_shards",
                IntegerAttr::get(IntegerType::get(ctx, 64), numShards));
    op->setAttr("nsp.scatter_grid", StringAttr::get(ctx, grid.getSymName()));
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

std::unique_ptr<Pass> createNSPShardPlannerPass(int64_t numCores,
                                                int64_t numThreads,
                                                bool allowCollectives) {
  ShardPolicy policy;
  policy.numCores = numCores;
  policy.numThreads = numThreads;
  policy.allowCollectives = allowCollectives;
  return std::make_unique<NSPShardPlannerPass>(policy);
}

std::unique_ptr<Pass> createNSPShardPlannerPass(int64_t nspCount,
                                                bool allowCollectives) {
  ShardPolicy policy;
  policy.numCores = nspCount;
  policy.numThreads = 1;
  policy.allowCollectives = allowCollectives;
  return std::make_unique<NSPShardPlannerPass>(policy);
}

} // namespace hexagon
} // namespace mlir
