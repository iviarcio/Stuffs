//===- NSPShardInterface.cpp - NSP Shard Interface Models -----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Register shard::ShardingInterface external models for a focused set of
// "boundary / view / sink / structural" operations that appear in the NSP
// sharding + SPMD pipeline.
//
// Rationale
// ---------
// shard-propagation depends on shard::ShardingInterface. If an operation does
// not implement it, propagation may stop at that boundary.
//
// This file covers non-structured ops that commonly appear around:
//   * View-like transformations
//   * Allocation / copy boundaries
//   * Bufferization boundaries (tensor <-> memref)
//   * Scalar / control-flow constructs
//   * Elementwise math/arith ops (e.g., math.exp, math.tanh)
//   * linalg.transpose / linalg.reduce sharding models
//
// Design Principles
// -----------------
// * Preserve existing sharding when semantically safe.
// * Treat boundaries as propagation boundaries, not compute semantics.
// * For elementwise ops, propagate sharding operand <-> result.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"

// Dialects / ops we model.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

// Shard interface.
#include "mlir/Dialect/Shard/Interfaces/ShardingInterface.h"
#include "mlir/Dialect/Shard/Interfaces/ShardingInterfaceImpl.h"

using namespace mlir;

namespace {

/// Returns a ranked shaped rank from the first ranked shaped operand/result.
/// Falls back to 0 if nothing is ranked (best-effort for propagation).
static int64_t getFirstRankedShapedRank(Operation *op) {
  auto getRank = [](Type t) -> std::optional<int64_t> {
    if (auto st = dyn_cast<ShapedType>(t)) {
      if (st.hasRank())
        return st.getRank();
    }
    return std::nullopt;
  };

  for (Type t : op->getOperandTypes())
    if (auto r = getRank(t))
      return *r;
  for (Type t : op->getResultTypes())
    if (auto r = getRank(t))
      return *r;

  return 0;
}

static bool hasRankedShapedType(Operation *op) {
  auto isRankedShaped = [](Type t) {
    auto st = dyn_cast<ShapedType>(t);
    return st && st.hasRank();
  };
  for (Type t : op->getOperandTypes())
    if (isRankedShaped(t))
      return true;
  for (Type t : op->getResultTypes())
    if (isRankedShaped(t))
      return true;
  return false;
}

static SmallVector<AffineMap> makeIdentityMapsForOp(Operation *op,
                                                    int64_t rank) {
  MLIRContext *ctx = op->getContext();
  AffineMap id = AffineMap::getMultiDimIdentityMap(rank, ctx);

  SmallVector<AffineMap> maps;
  maps.reserve(op->getNumOperands() + op->getNumResults());
  for (unsigned i = 0, e = op->getNumOperands() + op->getNumResults(); i < e;
       ++i)
    maps.push_back(id);
  return maps;
}

static SmallVector<utils::IteratorType> makeParallelIters(int64_t rank) {
  SmallVector<utils::IteratorType> iters;
  iters.reserve(rank);
  for (int64_t i = 0; i < rank; ++i)
    iters.push_back(utils::IteratorType::parallel);
  return iters;
}

/// Convert a Sharding's split-axes into a ShardingArray.
/// For these ops we treat each tensor dimension as an independent iterator.
static shard::ShardingArray toShardingArray(shard::Sharding sharding) {
  shard::ShardingArray arr;
  if (!sharding)
    return arr;

  arr.reserve(sharding.getSplitAxes().size());
  for (shard::GridAxesAttr axesAttr : sharding.getSplitAxes()) {
    SmallVector<shard::GridAxis> axes;
    auto ref = axesAttr.asArrayRef();
    axes.reserve(ref.size());
    for (int16_t a : ref)
      axes.push_back(a);
    arr.push_back(std::move(axes));
  }

  // Keep at least one sub-array to distinguish "not sharded" from "no info".
  shard::removeTrailingEmptySubArray(arr);
  return arr;
}

/// Build a Sharding from a ShardingOption by treating each shardingArray entry
/// as the split-axes for the corresponding tensor dimension.
static shard::Sharding fromShardingOption(Operation *op,
                                          const shard::ShardingOption &opt,
                                          int64_t rank) {
  if (!opt.grid)
    return shard::Sharding();

  SmallVector<shard::GridAxesAttr> splitAxes;
  splitAxes.reserve(rank);

  MLIRContext *ctx = op->getContext();
  ArrayRef<SmallVector<shard::GridAxis>> arr = opt.shardingArray;

  for (int64_t i = 0; i < rank; ++i) {
    SmallVector<int16_t> axesI16;
    if (i < (int64_t)arr.size()) {
      axesI16.reserve(arr[i].size());
      for (shard::GridAxis a : arr[i])
        axesI16.push_back(static_cast<int16_t>(a));
    }
    splitAxes.push_back(shard::GridAxesAttr::get(ctx, axesI16));
  }

  return shard::Sharding::get(opt.grid, splitAxes);
}

/// Create a ShardingOption corresponding to a specific value sharding.
static FailureOr<shard::ShardingOption>
makeValueShardingOption(shard::Sharding s) {
  if (!s)
    return shard::ShardingOption::makeEmpty();
  return shard::ShardingOption(toShardingArray(s), s.getGridAttr());
}

/// Build a ShardingOption directly from a split-axes array and grid.
static FailureOr<shard::ShardingOption>
makeShardingOptionFromAxes(ArrayRef<SmallVector<shard::GridAxis>> arr,
                           Attribute grid) {
  if (!grid)
    return shard::ShardingOption::makeEmpty();
  return shard::ShardingOption(arr, grid);
}

/// Convert split_axes into a fixed-size sharding array of rank `rank`.
static shard::ShardingArray toFixedRankShardingArray(shard::Sharding sharding,
                                                     int64_t rank) {
  shard::ShardingArray arr;
  arr.resize(rank);

  if (!sharding)
    return arr;

  auto splitAxes = sharding.getSplitAxes();
  for (int64_t i = 0; i < rank && i < (int64_t)splitAxes.size(); ++i) {
    for (int16_t a : splitAxes[i].asArrayRef())
      arr[i].push_back(a);
  }
  return arr;
}

/// Permute a sharding array according to a transpose permutation.
///
/// For linalg.transpose with permutation P:
///   result_dim[i] = input_dim[P[i]]
///
/// Therefore:
///   result_axes[i] = input_axes[P[i]]
static shard::ShardingArray
permuteShardingArray(ArrayRef<SmallVector<shard::GridAxis>> in,
                     ArrayRef<int64_t> permutation) {
  shard::ShardingArray out;
  out.resize(permutation.size());

  for (size_t i = 0; i < permutation.size(); ++i) {
    int64_t src = permutation[i];
    if (src >= 0 && src < (int64_t)in.size())
      out[i] = in[src];
  }
  return out;
}

/// Inverse permutation for transpose.
static shard::ShardingArray
inversePermuteShardingArray(ArrayRef<SmallVector<shard::GridAxis>> in,
                            ArrayRef<int64_t> permutation) {
  shard::ShardingArray out;
  out.resize(permutation.size());

  for (size_t i = 0; i < permutation.size(); ++i) {
    int64_t src = permutation[i];
    if (src >= 0 && src < (int64_t)out.size() && i < in.size())
      out[src] = in[i];
  }
  return out;
}

/// Project input split_axes through a linalg.reduce.
///
/// If the input is rank-N and the reduction removes dimensions in `dims`,
/// then the output keeps only the split_axes of the non-reduced dimensions,
/// preserving order.
static shard::ShardingArray
projectReduceInputToOutput(ArrayRef<SmallVector<shard::GridAxis>> in,
                           ArrayRef<int64_t> reductionDims) {
  llvm::SmallDenseSet<int64_t> reduced;
  for (int64_t d : reductionDims)
    reduced.insert(d);

  shard::ShardingArray out;
  for (int64_t i = 0; i < (int64_t)in.size(); ++i) {
    if (!reduced.contains(i))
      out.push_back(in[i]);
  }
  return out;
}

/// Expand output split_axes back to the input rank for linalg.reduce.
/// Note: Reduced dimensions are conservatively marked replicated.
static shard::ShardingArray
expandReduceOutputToInput(ArrayRef<SmallVector<shard::GridAxis>> out,
                          int64_t inputRank, ArrayRef<int64_t> reductionDims) {
  llvm::SmallDenseSet<int64_t> reduced;
  for (int64_t d : reductionDims)
    reduced.insert(d);

  shard::ShardingArray in;
  in.resize(inputRank);

  int64_t outIdx = 0;
  for (int64_t i = 0; i < inputRank; ++i) {
    if (reduced.contains(i))
      continue;
    if (outIdx < (int64_t)out.size())
      in[i] = out[outIdx++];
  }
  return in;
}

/// Partition helper: clone `op` with `partitionedOperands` and map results.
static LogicalResult partitionByCloning(Operation *op,
                                        ArrayRef<Value> partitionedOperands,
                                        IRMapping &partitionMap,
                                        OpBuilder &builder) {
  if (partitionedOperands.size() != op->getNumOperands())
    return failure();

  IRMapping localMap;
  for (auto [oldV, newV] : llvm::zip(op->getOperands(), partitionedOperands))
    localMap.map(oldV, newV);

  Operation *cloned = builder.clone(*op, localMap);
  for (auto [oldR, newR] : llvm::zip(op->getResults(), cloned->getResults()))
    partitionMap.map(oldR, newR);

  return success();
}

//===----------------------------------------------------------------------===//
// Generic models
//===----------------------------------------------------------------------===//

/// Generic model for scalar/control-flow ops that should not participate in
/// sharding. It tells shard-propagation to ignore the op instead of erroring
/// out.
template <typename OpTy>
struct NoShardingModel
    : public shard::ShardingInterface::ExternalModel<NoShardingModel<OpTy>,
                                                     OpTy> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *) const {
    return {};
  }
  SmallVector<shard::ReductionKind>
  getReductionLoopIteratorKinds(Operation *) const {
    return {};
  }
  SmallVector<AffineMap> getIndexingMaps(Operation *) const { return {}; }

  FailureOr<shard::ShardingOption>
  getShardingOption(Operation *, ArrayRef<shard::Sharding>,
                    ArrayRef<shard::Sharding>) const {
    return shard::ShardingOption::makeEmpty();
  }

  FailureOr<std::vector<shard::Sharding>>
  getShardingAnnotations(Operation *op, const shard::ShardingOption &) const {
    std::vector<shard::Sharding> res;
    res.resize(op->getNumOperands() + op->getNumResults(), shard::Sharding());
    return res;
  }

  LogicalResult addShardingAnnotations(Operation *, OpBuilder &,
                                       const shard::ShardingOption &) const {
    return success();
  }

  LogicalResult partition(Operation *op, ArrayRef<Value> partitionedOperands,
                          ArrayRef<shard::Sharding>, ArrayRef<shard::Sharding>,
                          IRMapping &partitionMap, SymbolTableCollection &,
                          OpBuilder &builder) const {
    return partitionByCloning(op, partitionedOperands, partitionMap, builder);
  }
};

/// Generic elementwise model for unary/binary arith/math ops.
/// For ranked shaped types, propagate sharding operand <-> result.
/// For pure scalars, behave like NoShardingModel (empty info).
template <typename OpTy>
struct ElementwiseShardingModel
    : public shard::ShardingInterface::ExternalModel<
          ElementwiseShardingModel<OpTy>, OpTy> {

  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    if (!hasRankedShapedType(op))
      return {};
    int64_t rank = getFirstRankedShapedRank(op);
    return makeIdentityMapsForOp(op, rank);
  }

  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    if (!hasRankedShapedType(op))
      return {};
    int64_t rank = getFirstRankedShapedRank(op);
    return makeParallelIters(rank);
  }

  SmallVector<shard::ReductionKind>
  getReductionLoopIteratorKinds(Operation *) const {
    return {};
  }

  FailureOr<shard::ShardingOption>
  getShardingOption(Operation *op, ArrayRef<shard::Sharding> operandShardings,
                    ArrayRef<shard::Sharding> resultShardings) const {
    // Scalar-only instances: no sharding info.
    if (!hasRankedShapedType(op))
      return shard::ShardingOption::makeEmpty();

    // Prefer the first known operand sharding.
    for (shard::Sharding s : operandShardings)
      if (s)
        return makeValueShardingOption(s);

    // Otherwise accept a proposed result sharding.
    for (shard::Sharding s : resultShardings)
      if (s)
        return makeValueShardingOption(s);

    return shard::ShardingOption::makeEmpty();
  }

  FailureOr<std::vector<shard::Sharding>>
  getShardingAnnotations(Operation *op,
                         const shard::ShardingOption &opt) const {
    std::vector<shard::Sharding> res;
    res.resize(op->getNumOperands() + op->getNumResults(), shard::Sharding());

    if (!hasRankedShapedType(op))
      return res;

    int64_t rank = getFirstRankedShapedRank(op);
    shard::Sharding s = fromShardingOption(op, opt, rank);

    for (auto &x : res)
      x = s;
    return res;
  }

  LogicalResult addShardingAnnotations(Operation *, OpBuilder &,
                                       const shard::ShardingOption &) const {
    // Elementwise ops do not need explicit shard ops inserted.
    return success();
  }

  LogicalResult partition(Operation *op, ArrayRef<Value> partitionedOperands,
                          ArrayRef<shard::Sharding>, ArrayRef<shard::Sharding>,
                          IRMapping &partitionMap, SymbolTableCollection &,
                          OpBuilder &builder) const {
    return partitionByCloning(op, partitionedOperands, partitionMap, builder);
  }
};

/// Result-only model for ops with 0 operands and (typically) 1 result.
/// Useful for tensor constants: accept a proposed result sharding when present.
template <typename OpTy>
struct ResultOnlyShardingModel : public shard::ShardingInterface::ExternalModel<
                                     ResultOnlyShardingModel<OpTy>, OpTy> {

  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    if (!hasRankedShapedType(op))
      return {};
    int64_t rank = getFirstRankedShapedRank(op);
    return makeIdentityMapsForOp(op, rank);
  }

  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    if (!hasRankedShapedType(op))
      return {};
    int64_t rank = getFirstRankedShapedRank(op);
    return makeParallelIters(rank);
  }

  SmallVector<shard::ReductionKind>
  getReductionLoopIteratorKinds(Operation *) const {
    return {};
  }

  FailureOr<shard::ShardingOption>
  getShardingOption(Operation *op, ArrayRef<shard::Sharding>,
                    ArrayRef<shard::Sharding> resultShardings) const {
    if (!hasRankedShapedType(op))
      return shard::ShardingOption::makeEmpty();

    if (!resultShardings.empty() && resultShardings[0])
      return makeValueShardingOption(resultShardings[0]);

    return shard::ShardingOption::makeEmpty();
  }

  FailureOr<std::vector<shard::Sharding>>
  getShardingAnnotations(Operation *op,
                         const shard::ShardingOption &opt) const {
    std::vector<shard::Sharding> res;
    res.resize(op->getNumOperands() + op->getNumResults(), shard::Sharding());

    if (!hasRankedShapedType(op))
      return res;

    int64_t rank = getFirstRankedShapedRank(op);
    shard::Sharding s = fromShardingOption(op, opt, rank);
    for (auto &x : res)
      x = s;
    return res;
  }

  LogicalResult addShardingAnnotations(Operation *, OpBuilder &,
                                       const shard::ShardingOption &) const {
    return success();
  }

  LogicalResult partition(Operation *op, ArrayRef<Value> partitionedOperands,
                          ArrayRef<shard::Sharding>, ArrayRef<shard::Sharding>,
                          IRMapping &partitionMap, SymbolTableCollection &,
                          OpBuilder &builder) const {
    return partitionByCloning(op, partitionedOperands, partitionMap, builder);
  }
};

//===----------------------------------------------------------------------===//
// memref.reinterpret_cast / memref.alloc / memref.copy
//===----------------------------------------------------------------------===//

struct ReinterpretCastShardingModel
    : public shard::ShardingInterface::ExternalModel<
          ReinterpretCastShardingModel, memref::ReinterpretCastOp> {
  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    int64_t rank = getFirstRankedShapedRank(op);
    return makeIdentityMapsForOp(op, rank);
  }
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    int64_t rank = getFirstRankedShapedRank(op);
    return makeParallelIters(rank);
  }
  SmallVector<shard::ReductionKind>
  getReductionLoopIteratorKinds(Operation *) const {
    return {};
  }

  FailureOr<shard::ShardingOption>
  getShardingOption(Operation *op, ArrayRef<shard::Sharding> operandShardings,
                    ArrayRef<shard::Sharding> resultShardings) const {
    (void)resultShardings;
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return failure();

    if (!operandShardings.empty() && operandShardings[0])
      return makeValueShardingOption(operandShardings[0]);

    return shard::ShardingOption::makeEmpty();
  }

  FailureOr<std::vector<shard::Sharding>>
  getShardingAnnotations(Operation *op,
                         const shard::ShardingOption &opt) const {
    int64_t rank = getFirstRankedShapedRank(op);
    shard::Sharding s = fromShardingOption(op, opt, rank);

    std::vector<shard::Sharding> res;
    res.reserve(2);
    res.push_back(s);
    res.push_back(s);
    return res;
  }

  LogicalResult addShardingAnnotations(Operation *, OpBuilder &,
                                       const shard::ShardingOption &) const {
    return success();
  }

  LogicalResult partition(Operation *op, ArrayRef<Value> partitionedOperands,
                          ArrayRef<shard::Sharding>, ArrayRef<shard::Sharding>,
                          IRMapping &partitionMap, SymbolTableCollection &,
                          OpBuilder &builder) const {
    return partitionByCloning(op, partitionedOperands, partitionMap, builder);
  }
};

struct AllocShardingModel
    : public shard::ShardingInterface::ExternalModel<AllocShardingModel,
                                                     memref::AllocOp> {
  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    int64_t rank = getFirstRankedShapedRank(op);
    return makeIdentityMapsForOp(op, rank);
  }
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    int64_t rank = getFirstRankedShapedRank(op);
    return makeParallelIters(rank);
  }
  SmallVector<shard::ReductionKind>
  getReductionLoopIteratorKinds(Operation *) const {
    return {};
  }

  FailureOr<shard::ShardingOption>
  getShardingOption(Operation *op, ArrayRef<shard::Sharding>,
                    ArrayRef<shard::Sharding> resultShardings) const {
    if (op->getNumOperands() != 0 || op->getNumResults() != 1)
      return failure();

    if (!resultShardings.empty() && resultShardings[0])
      return makeValueShardingOption(resultShardings[0]);

    return shard::ShardingOption::makeEmpty();
  }

  FailureOr<std::vector<shard::Sharding>>
  getShardingAnnotations(Operation *op,
                         const shard::ShardingOption &opt) const {
    int64_t rank = getFirstRankedShapedRank(op);
    shard::Sharding s = fromShardingOption(op, opt, rank);

    std::vector<shard::Sharding> res;
    res.reserve(1);
    res.push_back(s);
    return res;
  }

  LogicalResult addShardingAnnotations(Operation *, OpBuilder &,
                                       const shard::ShardingOption &) const {
    return success();
  }

  LogicalResult partition(Operation *op, ArrayRef<Value> partitionedOperands,
                          ArrayRef<shard::Sharding>, ArrayRef<shard::Sharding>,
                          IRMapping &partitionMap, SymbolTableCollection &,
                          OpBuilder &builder) const {
    return partitionByCloning(op, partitionedOperands, partitionMap, builder);
  }
};

struct CopyShardingModel
    : public shard::ShardingInterface::ExternalModel<CopyShardingModel,
                                                     memref::CopyOp> {
  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    int64_t rank = getFirstRankedShapedRank(op);
    return makeIdentityMapsForOp(op, rank);
  }
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    int64_t rank = getFirstRankedShapedRank(op);
    return makeParallelIters(rank);
  }
  SmallVector<shard::ReductionKind>
  getReductionLoopIteratorKinds(Operation *) const {
    return {};
  }

  FailureOr<shard::ShardingOption>
  getShardingOption(Operation *op, ArrayRef<shard::Sharding> operandShardings,
                    ArrayRef<shard::Sharding>) const {
    if (op->getNumOperands() != 2 || op->getNumResults() != 0)
      return failure();

    // Prefer src sharding; otherwise accept dst sharding.
    if (operandShardings.size() >= 1 && operandShardings[0])
      return makeValueShardingOption(operandShardings[0]);
    if (operandShardings.size() >= 2 && operandShardings[1])
      return makeValueShardingOption(operandShardings[1]);

    return shard::ShardingOption::makeEmpty();
  }

  FailureOr<std::vector<shard::Sharding>>
  getShardingAnnotations(Operation *op,
                         const shard::ShardingOption &opt) const {
    int64_t rank = getFirstRankedShapedRank(op);
    shard::Sharding s = fromShardingOption(op, opt, rank);

    std::vector<shard::Sharding> res;
    res.reserve(2);
    res.push_back(s); // src
    res.push_back(s); // dst
    return res;
  }

  LogicalResult addShardingAnnotations(Operation *, OpBuilder &,
                                       const shard::ShardingOption &) const {
    return success();
  }

  LogicalResult partition(Operation *op, ArrayRef<Value> partitionedOperands,
                          ArrayRef<shard::Sharding>, ArrayRef<shard::Sharding>,
                          IRMapping &partitionMap, SymbolTableCollection &,
                          OpBuilder &builder) const {
    return partitionByCloning(op, partitionedOperands, partitionMap, builder);
  }
};

//===----------------------------------------------------------------------===//
// bufferization.alloc_tensor / bufferization.to_tensor /
// bufferization.to_buffer / bufferization.materialize_in_destination
//===----------------------------------------------------------------------===//

struct AllocTensorShardingModel
    : public shard::ShardingInterface::ExternalModel<
          AllocTensorShardingModel, bufferization::AllocTensorOp> {

  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    int64_t rank = getFirstRankedShapedRank(op);
    return makeIdentityMapsForOp(op, rank);
  }

  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    int64_t rank = getFirstRankedShapedRank(op);
    return makeParallelIters(rank);
  }

  SmallVector<shard::ReductionKind>
  getReductionLoopIteratorKinds(Operation *) const {
    return {};
  }

  FailureOr<shard::ShardingOption>
  getShardingOption(Operation *op, ArrayRef<shard::Sharding> operandShardings,
                    ArrayRef<shard::Sharding> resultShardings) const {

    auto alloc = cast<bufferization::AllocTensorOp>(op);

    // If alloc_tensor copies from another tensor, inherit its sharding.
    if (Value copy = alloc.getCopy()) {
      for (auto [idx, operand] : llvm::enumerate(op->getOperands())) {
        if (operand == copy) {
          if (idx < operandShardings.size() && operandShardings[idx])
            return makeValueShardingOption(operandShardings[idx]);
          break;
        }
      }
    }

    // Otherwise accept proposed result sharding.
    if (!resultShardings.empty() && resultShardings[0])
      return makeValueShardingOption(resultShardings[0]);

    return shard::ShardingOption::makeEmpty();
  }

  FailureOr<std::vector<shard::Sharding>>
  getShardingAnnotations(Operation *op,
                         const shard::ShardingOption &opt) const {

    int64_t rank = getFirstRankedShapedRank(op);
    shard::Sharding s = fromShardingOption(op, opt, rank);

    std::vector<shard::Sharding> res;
    res.resize(op->getNumOperands() + op->getNumResults(), shard::Sharding());

    // Only annotate the result tensor.
    if (!res.empty())
      res.back() = s;

    return res;
  }

  LogicalResult addShardingAnnotations(Operation *, OpBuilder &,
                                       const shard::ShardingOption &) const {
    return success();
  }

  LogicalResult partition(Operation *op, ArrayRef<Value> partitionedOperands,
                          ArrayRef<shard::Sharding>, ArrayRef<shard::Sharding>,
                          IRMapping &partitionMap, SymbolTableCollection &,
                          OpBuilder &builder) const {
    return partitionByCloning(op, partitionedOperands, partitionMap, builder);
  }
};

struct ToTensorShardingModel
    : public shard::ShardingInterface::ExternalModel<
          ToTensorShardingModel, bufferization::ToTensorOp> {
  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    int64_t rank = getFirstRankedShapedRank(op);
    return makeIdentityMapsForOp(op, rank);
  }
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    int64_t rank = getFirstRankedShapedRank(op);
    return makeParallelIters(rank);
  }
  SmallVector<shard::ReductionKind>
  getReductionLoopIteratorKinds(Operation *) const {
    return {};
  }

  FailureOr<shard::ShardingOption>
  getShardingOption(Operation *op, ArrayRef<shard::Sharding>,
                    ArrayRef<shard::Sharding> resultShardings) const {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return failure();

    if (!resultShardings.empty() && resultShardings[0])
      return makeValueShardingOption(resultShardings[0]);

    return shard::ShardingOption::makeEmpty();
  }

  FailureOr<std::vector<shard::Sharding>>
  getShardingAnnotations(Operation *op,
                         const shard::ShardingOption &opt) const {
    int64_t rank = getFirstRankedShapedRank(op);
    shard::Sharding s = fromShardingOption(op, opt, rank);

    std::vector<shard::Sharding> res;
    res.reserve(2);
    res.push_back(s);
    res.push_back(s);
    return res;
  }

  LogicalResult addShardingAnnotations(Operation *, OpBuilder &,
                                       const shard::ShardingOption &) const {
    return success();
  }

  LogicalResult partition(Operation *op, ArrayRef<Value> partitionedOperands,
                          ArrayRef<shard::Sharding>, ArrayRef<shard::Sharding>,
                          IRMapping &partitionMap, SymbolTableCollection &,
                          OpBuilder &builder) const {
    return partitionByCloning(op, partitionedOperands, partitionMap, builder);
  }
};

struct ToBufferShardingModel
    : public shard::ShardingInterface::ExternalModel<
          ToBufferShardingModel, bufferization::ToBufferOp> {
  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    int64_t rank = getFirstRankedShapedRank(op);
    return makeIdentityMapsForOp(op, rank);
  }
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    int64_t rank = getFirstRankedShapedRank(op);
    return makeParallelIters(rank);
  }
  SmallVector<shard::ReductionKind>
  getReductionLoopIteratorKinds(Operation *) const {
    return {};
  }

  FailureOr<shard::ShardingOption>
  getShardingOption(Operation *op, ArrayRef<shard::Sharding>,
                    ArrayRef<shard::Sharding> resultShardings) const {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return failure();

    if (!resultShardings.empty() && resultShardings[0])
      return makeValueShardingOption(resultShardings[0]);

    return shard::ShardingOption::makeEmpty();
  }

  FailureOr<std::vector<shard::Sharding>>
  getShardingAnnotations(Operation *op,
                         const shard::ShardingOption &opt) const {
    int64_t rank = getFirstRankedShapedRank(op);
    shard::Sharding s = fromShardingOption(op, opt, rank);

    std::vector<shard::Sharding> res;
    res.reserve(2);
    res.push_back(s);
    res.push_back(s);
    return res;
  }

  LogicalResult addShardingAnnotations(Operation *, OpBuilder &,
                                       const shard::ShardingOption &) const {
    return success();
  }

  LogicalResult partition(Operation *op, ArrayRef<Value> partitionedOperands,
                          ArrayRef<shard::Sharding>, ArrayRef<shard::Sharding>,
                          IRMapping &partitionMap, SymbolTableCollection &,
                          OpBuilder &builder) const {
    return partitionByCloning(op, partitionedOperands, partitionMap, builder);
  }
};

struct MaterializeInDestShardingModel
    : public shard::ShardingInterface::ExternalModel<
          MaterializeInDestShardingModel,
          bufferization::MaterializeInDestinationOp> {
  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    int64_t rank = getFirstRankedShapedRank(op);
    return makeIdentityMapsForOp(op, rank);
  }
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    int64_t rank = getFirstRankedShapedRank(op);
    return makeParallelIters(rank);
  }
  SmallVector<shard::ReductionKind>
  getReductionLoopIteratorKinds(Operation *) const {
    return {};
  }

  FailureOr<shard::ShardingOption>
  getShardingOption(Operation *op, ArrayRef<shard::Sharding> operandShardings,
                    ArrayRef<shard::Sharding>) const {
    if (op->getNumOperands() != 2 || op->getNumResults() != 0)
      return failure();

    // Operand(0) is the tensor to materialize. If it has sharding, accept it.
    if (!operandShardings.empty() && operandShardings[0])
      return makeValueShardingOption(operandShardings[0]);

    return shard::ShardingOption::makeEmpty();
  }

  FailureOr<std::vector<shard::Sharding>>
  getShardingAnnotations(Operation *op,
                         const shard::ShardingOption &opt) const {
    int64_t rank = getFirstRankedShapedRank(op);
    shard::Sharding s = fromShardingOption(op, opt, rank);

    std::vector<shard::Sharding> res;
    res.resize(op->getNumOperands() + op->getNumResults(), shard::Sharding());
    // Both operands get the same logical sharding info (tensor + destination
    // memref).
    if (res.size() >= 2) {
      res[0] = s;
      res[1] = s;
    }
    return res;
  }

  LogicalResult addShardingAnnotations(Operation *, OpBuilder &,
                                       const shard::ShardingOption &) const {
    return success();
  }

  LogicalResult partition(Operation *op, ArrayRef<Value> partitionedOperands,
                          ArrayRef<shard::Sharding>, ArrayRef<shard::Sharding>,
                          IRMapping &partitionMap, SymbolTableCollection &,
                          OpBuilder &builder) const {
    return partitionByCloning(op, partitionedOperands, partitionMap, builder);
  }
};

//===----------------------------------------------------------------------===//
// linalg.transpose / linalg.reduce
//===----------------------------------------------------------------------===//

struct TransposeShardingModel
    : public shard::ShardingInterface::ExternalModel<TransposeShardingModel,
                                                     linalg::TransposeOp> {
  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    auto transpose = cast<linalg::TransposeOp>(op);
    auto inputTy = dyn_cast<ShapedType>(transpose.getInput().getType());
    auto outputTy = dyn_cast<ShapedType>(transpose.getInit().getType());
    if (!inputTy || !outputTy || !inputTy.hasRank() || !outputTy.hasRank())
      return {};

    MLIRContext *ctx = op->getContext();
    int64_t inRank = inputTy.getRank();
    int64_t outRank = outputTy.getRank();

    SmallVector<AffineExpr> inExprs;
    SmallVector<AffineExpr> outExprs;
    inExprs.reserve(inRank);
    outExprs.reserve(outRank);

    for (int64_t i = 0; i < inRank; ++i)
      inExprs.push_back(mlir::getAffineDimExpr(i, ctx));

    auto perm = transpose.getPermutation();
    for (int64_t i = 0; i < outRank; ++i)
      outExprs.push_back(mlir::getAffineDimExpr(perm[i], ctx));

    return {
        AffineMap::get(inRank, 0, inExprs, ctx),
        AffineMap::get(inRank, 0, outExprs, ctx),
    };
  }

  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto transpose = cast<linalg::TransposeOp>(op);
    auto inputTy = dyn_cast<ShapedType>(transpose.getInput().getType());
    if (!inputTy || !inputTy.hasRank())
      return {};
    return makeParallelIters(inputTy.getRank());
  }

  SmallVector<shard::ReductionKind>
  getReductionLoopIteratorKinds(Operation *) const {
    return {};
  }

  FailureOr<shard::ShardingOption>
  getShardingOption(Operation *op, ArrayRef<shard::Sharding> operandShardings,
                    ArrayRef<shard::Sharding> resultShardings) const {
    auto transpose = cast<linalg::TransposeOp>(op);
    auto perm = transpose.getPermutation();

    if (!operandShardings.empty() && operandShardings[0]) {
      auto arr = toFixedRankShardingArray(
          operandShardings[0],
          cast<ShapedType>(transpose.getInput().getType()).getRank());
      return makeShardingOptionFromAxes(permuteShardingArray(arr, perm),
                                        operandShardings[0].getGridAttr());
    }

    if (!resultShardings.empty() && resultShardings[0]) {
      auto arr = toFixedRankShardingArray(
          resultShardings[0],
          cast<ShapedType>(transpose.getResult()[0].getType()).getRank());
      return makeShardingOptionFromAxes(inversePermuteShardingArray(arr, perm),
                                        resultShardings[0].getGridAttr());
    }

    return shard::ShardingOption::makeEmpty();
  }

  FailureOr<std::vector<shard::Sharding>>
  getShardingAnnotations(Operation *op,
                         const shard::ShardingOption &opt) const {
    auto transpose = cast<linalg::TransposeOp>(op);
    auto inputTy = cast<ShapedType>(transpose.getInput().getType());
    auto resultTy = cast<ShapedType>(transpose.getResult()[0].getType());

    std::vector<shard::Sharding> res;
    res.resize(op->getNumOperands() + op->getNumResults(), shard::Sharding());

    shard::Sharding resultSharding =
        fromShardingOption(op, opt, resultTy.getRank());
    res[1] = resultSharding;

    auto resultArr =
        toFixedRankShardingArray(resultSharding, resultTy.getRank());
    auto inputArr =
        inversePermuteShardingArray(resultArr, transpose.getPermutation());
    auto inputOpt =
        shard::ShardingOption(inputArr, resultSharding.getGridAttr());
    res[0] = fromShardingOption(op, inputOpt, inputTy.getRank());

    return res;
  }

  LogicalResult addShardingAnnotations(Operation *, OpBuilder &,
                                       const shard::ShardingOption &) const {
    return success();
  }

  LogicalResult partition(Operation *op, ArrayRef<Value> partitionedOperands,
                          ArrayRef<shard::Sharding>, ArrayRef<shard::Sharding>,
                          IRMapping &partitionMap, SymbolTableCollection &,
                          OpBuilder &builder) const {
    return partitionByCloning(op, partitionedOperands, partitionMap, builder);
  }
};

struct ReduceShardingModel
    : public shard::ShardingInterface::ExternalModel<ReduceShardingModel,
                                                     linalg::ReduceOp> {
  SmallVector<AffineMap> getIndexingMaps(Operation *) const { return {}; }

  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *) const {
    return {};
  }

  SmallVector<shard::ReductionKind>
  getReductionLoopIteratorKinds(Operation *) const {
    return {};
  }

  FailureOr<shard::ShardingOption>
  getShardingOption(Operation *op, ArrayRef<shard::Sharding> operandShardings,
                    ArrayRef<shard::Sharding> resultShardings) const {
    auto reduce = cast<linalg::ReduceOp>(op);
    auto inputTy = dyn_cast<ShapedType>(reduce.getInputs()[0].getType());
    auto resultTy = dyn_cast<ShapedType>(reduce.getResults()[0].getType());
    if (!inputTy || !resultTy || !inputTy.hasRank() || !resultTy.hasRank())
      return shard::ShardingOption::makeEmpty();

    SmallVector<int64_t> dims;
    for (int64_t d : reduce.getDimensions())
      dims.push_back(d);

    if (!operandShardings.empty() && operandShardings[0]) {
      auto inArr =
          toFixedRankShardingArray(operandShardings[0], inputTy.getRank());
      auto outArr = projectReduceInputToOutput(inArr, dims);
      return makeShardingOptionFromAxes(outArr,
                                        operandShardings[0].getGridAttr());
    }

    if (!resultShardings.empty() && resultShardings[0]) {
      auto outArr =
          toFixedRankShardingArray(resultShardings[0], resultTy.getRank());
      auto inArr = expandReduceOutputToInput(outArr, inputTy.getRank(), dims);
      return makeShardingOptionFromAxes(inArr,
                                        resultShardings[0].getGridAttr());
    }

    return shard::ShardingOption::makeEmpty();
  }

  FailureOr<std::vector<shard::Sharding>>
  getShardingAnnotations(Operation *op,
                         const shard::ShardingOption &opt) const {
    std::vector<shard::Sharding> res;
    res.resize(op->getNumOperands() + op->getNumResults(), shard::Sharding());

    auto reduce = cast<linalg::ReduceOp>(op);
    auto inputTy = cast<ShapedType>(reduce.getInputs()[0].getType());
    auto resultTy = cast<ShapedType>(reduce.getResults()[0].getType());
    SmallVector<int64_t> dims;
    for (int64_t d : reduce.getDimensions())
      dims.push_back(d);

    shard::Sharding inputSharding =
        fromShardingOption(op, opt, inputTy.getRank());
    res[0] = inputSharding;

    auto inArr = toFixedRankShardingArray(inputSharding, inputTy.getRank());
    auto outArr = projectReduceInputToOutput(inArr, dims);
    auto outOpt = shard::ShardingOption(outArr, inputSharding.getGridAttr());
    res[1] = fromShardingOption(op, outOpt, resultTy.getRank());

    return res;
  }

  LogicalResult addShardingAnnotations(Operation *, OpBuilder &,
                                       const shard::ShardingOption &) const {
    return success();
  }

  LogicalResult partition(Operation *op, ArrayRef<Value> partitionedOperands,
                          ArrayRef<shard::Sharding>, ArrayRef<shard::Sharding>,
                          IRMapping &partitionMap, SymbolTableCollection &,
                          OpBuilder &builder) const {
    return partitionByCloning(op, partitionedOperands, partitionMap, builder);
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace hexagon {

/// Register all external sharding interface models used by the NSP pipeline.
/// This must be called during compiler initialization (DialectRegistry setup).
void registerNSPShardInterfaceModels(DialectRegistry &registry) {
  // Memref boundaries / structural ops.
  registry.addExtension(+[](MLIRContext *ctx, memref::MemRefDialect *dialect) {
    (void)dialect;
    memref::ReinterpretCastOp::attachInterface<ReinterpretCastShardingModel>(
        *ctx);
    memref::AllocOp::attachInterface<AllocShardingModel>(*ctx);
    memref::CopyOp::attachInterface<CopyShardingModel>(*ctx);

    // Pointer/metadata extraction: sharding-transparent.
    memref::ExtractAlignedPointerAsIndexOp::attachInterface<
        NoShardingModel<memref::ExtractAlignedPointerAsIndexOp>>(*ctx);
  });

  // Bufferization boundaries.
  registry.addExtension(
      +[](MLIRContext *ctx, bufferization::BufferizationDialect *dialect) {
        (void)dialect;
        bufferization::AllocTensorOp::attachInterface<AllocTensorShardingModel>(
            *ctx);
        bufferization::ToTensorOp::attachInterface<ToTensorShardingModel>(*ctx);
        bufferization::ToBufferOp::attachInterface<ToBufferShardingModel>(*ctx);
        bufferization::MaterializeInDestinationOp::attachInterface<
            MaterializeInDestShardingModel>(*ctx);
      });

  // Linalg structural ops (Transpose & Reduce)
  registry.addExtension(+[](MLIRContext *ctx, linalg::LinalgDialect *dialect) {
    (void)dialect;
    linalg::TransposeOp::attachInterface<TransposeShardingModel>(*ctx);
    linalg::ReduceOp::attachInterface<ReduceShardingModel>(*ctx);
  });

  // Control-flow (modeled as sharding-transparent for now).
  registry.addExtension(+[](MLIRContext *ctx, scf::SCFDialect *dialect) {
    (void)dialect;
    scf::ForOp::attachInterface<NoShardingModel<scf::ForOp>>(*ctx);
    scf::YieldOp::attachInterface<NoShardingModel<scf::YieldOp>>(*ctx);
  });

  // Arith: mix of scalar and tensor elementwise ops.
  registry.addExtension(+[](MLIRContext *ctx, arith::ArithDialect *dialect) {
    (void)dialect;

    // Constants may produce tensors; accept proposed result sharding.
    arith::ConstantOp::attachInterface<
        ResultOnlyShardingModel<arith::ConstantOp>>(*ctx);

    // Casts / loop-bound plumbing ops (scalar/control oriented).
    arith::IndexCastOp::attachInterface<NoShardingModel<arith::IndexCastOp>>(
        *ctx);

    // Elementwise (works for tensors; becomes "empty" for scalars).
    arith::AddFOp::attachInterface<ElementwiseShardingModel<arith::AddFOp>>(
        *ctx);
    arith::SubFOp::attachInterface<ElementwiseShardingModel<arith::SubFOp>>(
        *ctx);
    arith::MulFOp::attachInterface<ElementwiseShardingModel<arith::MulFOp>>(
        *ctx);
    arith::DivFOp::attachInterface<ElementwiseShardingModel<arith::DivFOp>>(
        *ctx);

    arith::AddIOp::attachInterface<ElementwiseShardingModel<arith::AddIOp>>(
        *ctx);
    arith::SubIOp::attachInterface<ElementwiseShardingModel<arith::SubIOp>>(
        *ctx);
    arith::MulIOp::attachInterface<ElementwiseShardingModel<arith::MulIOp>>(
        *ctx);
  });

  // Math: Commonly used ops: exp/tanh/erf
  registry.addExtension(+[](MLIRContext *ctx, math::MathDialect *dialect) {
    (void)dialect;
    math::ExpOp::attachInterface<ElementwiseShardingModel<math::ExpOp>>(*ctx);
    math::TanhOp::attachInterface<ElementwiseShardingModel<math::TanhOp>>(*ctx);
    math::ErfOp::attachInterface<ElementwiseShardingModel<math::ErfOp>>(*ctx);
  });

  // LLVM dialect ops that should be ignored by sharding propagation.
  registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
    (void)dialect;

    // External/runtime call boundary: do not propagate sharding through it.
    LLVM::CallOp::attachInterface<NoShardingModel<LLVM::CallOp>>(*ctx);

    // Pointer/Integer conversion ops: metadata/address plumbing only
    LLVM::IntToPtrOp::attachInterface<NoShardingModel<LLVM::IntToPtrOp>>(*ctx);
    LLVM::PtrToIntOp::attachInterface<NoShardingModel<LLVM::PtrToIntOp>>(*ctx);
  });
}

} // namespace hexagon
} // namespace mlir
