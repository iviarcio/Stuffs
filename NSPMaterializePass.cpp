//===- NSPMaterializePass.cpp - NSP destination materialization -----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//
//
// NSP materialization pass.
//
// This pass consumes the explicit `nsp.materialize_tile` hand-off operation
// emitted by NSPLocalizePass in non-collective mode.
//
// For each `nsp.materialize_tile %tile into %dest grid @g ...`, the pass:
//   1. computes the participant-local tile offset using
//      shard.process_linear_index,
//   2. creates a memref.subview into the final destination buffer,
//   3. tries a path that rewrites a localized tensor linalg.generic
//      into a memref-semantics linalg.generic writing directly into memref
//      inputs/outputs,
//   4. falls back to bufferization.materialize_in_destination when this
//      path is not applicable,
//   5. erases the temporary NSP hand-off op.
//
// Bring-up constraints:
//   - the generic tile hand-off supports rank-1 / rank-2 destination
//     materialization through tile_shape,
//   - split_axis is currently restricted to 0,
//   - the direct memref fast-paths remain rank-1 only,
//   - rank-2 currently falls back to bufferization.materialize_in_destination
//     unless a dedicated fast-path is added,
//   - expects a valid shard.grid symbol referenced by the NSP op.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shard/IR/ShardDialect.h"
#include "mlir/Dialect/Shard/IR/ShardOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "hexagon/Dialect/NSP/IR/NSPDialect.h"
#include "hexagon/Dialect/NSP/IR/NSPOps.h"

namespace mlir {
namespace hexagon {

namespace {

static FailureOr<mlir::shard::GridOp> lookupGrid(Operation *op,
                                                 FlatSymbolRefAttr gridAttr) {
  if (!gridAttr)
    return failure();

  auto grid =
      SymbolTable::lookupNearestSymbolFrom<mlir::shard::GridOp>(op, gridAttr);
  if (!grid)
    return failure();

  return grid;
}

/// Helper to extract tileShape
static FailureOr<SmallVector<int64_t>>
getStaticTileShape(mlir::hexagon::nsp::MaterializeTileOp tileOp) {
  auto tileShapeAttr = tileOp.getTileShapeAttr();
  if (!tileShapeAttr)
    return failure();

  SmallVector<int64_t> shape;
  shape.reserve(tileShapeAttr.size());
  for (Attribute attr : tileShapeAttr) {
    auto intAttr = dyn_cast<IntegerAttr>(attr);
    if (!intAttr)
      return failure();

    int64_t dim = intAttr.getInt();
    if (dim <= 0)
      return failure();

    shape.push_back(dim);
  }

  return shape;
}

/// Return true iff `g` is a localized elementwise generic that can be
/// safely rewritten from tensor semantics to memref semantics.
///
/// Current pattern constraints:
///   - carries the `nsp.localized` marker
///   - exactly 1 or 2 DPS inputs
///   - exactly 1 DPS init/output
///   - rank-1 elementwise shape
///   - one parallel iterator
///   - identity indexing maps
///
/// Note: the generic tile materialization path itself already accepts rank-2
/// tile_shape/destSubview, but this direct memref rewrite helper is still
/// intentionally rank-1 only.
static bool isSimpleLocalizedElementwiseGeneric(linalg::GenericOp g) {
  if (!g || !g->hasAttr("nsp.localized"))
    return false;

  const int64_t numInputs = g.getNumDpsInputs();
  if ((numInputs != 1 && numInputs != 2) || g.getNumDpsInits() != 1)
    return false;

  if (g.getNumLoops() != 1)
    return false;

  auto iters = g.getIteratorTypesArray();
  if (iters.size() != 1 || iters.front() != mlir::utils::IteratorType::parallel)
    return false;

  auto maps = g.getIndexingMapsArray();
  if (static_cast<int64_t>(maps.size()) != numInputs + 1)
    return false;

  for (AffineMap m : maps)
    if (!m.isIdentity())
      return false;

  auto resultTy = dyn_cast<RankedTensorType>(g.getResult(0).getType());
  if (!resultTy || resultTy.getRank() != 1)
    return false;

  for (OpOperand *in : g.getDpsInputOperands()) {
    auto inTy = dyn_cast<RankedTensorType>(in->get().getType());
    if (!inTy || inTy.getRank() != 1)
      return false;
  }

  return true;
}

static FailureOr<Value> stripTensorCasts(Value v) {
  Value cur = v;
  while (auto castOp = cur.getDefiningOp<tensor::CastOp>())
    cur = castOp.getSource();
  return cur;
}

/// Build a rank-2 memref.subview at the current insertion point.
static FailureOr<Value> buildRank2Subview(OpBuilder &b, Location loc,
                                          Value baseMemref,
                                          ArrayRef<OpFoldResult> offsets,
                                          ArrayRef<OpFoldResult> sizes,
                                          ArrayRef<OpFoldResult> strides) {
  auto baseTy = dyn_cast<MemRefType>(baseMemref.getType());
  if (!baseTy || baseTy.getRank() != 2)
    return failure();
  if (offsets.size() != 2 || sizes.size() != 2 || strides.size() != 2)
    return failure();

  int64_t staticSize0 = ShapedType::kDynamic;
  int64_t staticSize1 = ShapedType::kDynamic;

  if (auto a0 = dyn_cast<Attribute>(sizes[0]))
    if (auto i0 = dyn_cast<IntegerAttr>(a0))
      staticSize0 = i0.getInt();
  if (auto a1 = dyn_cast<Attribute>(sizes[1]))
    if (auto i1 = dyn_cast<IntegerAttr>(a1))
      staticSize1 = i1.getInt();

  auto subLayout = StridedLayoutAttr::get(
      baseTy.getContext(),
      /*offset=*/ShapedType::kDynamic,
      /*strides=*/ArrayRef<int64_t>{ShapedType::kDynamic, 1});

  auto subviewTy =
      MemRefType::get({staticSize0, staticSize1}, baseTy.getElementType(),
                      subLayout, baseTy.getMemorySpace());

  return b
      .create<memref::SubViewOp>(loc, subviewTy, baseMemref, offsets, sizes,
                                 strides)
      .getResult();
}

/// Match a rank-2 tensor.extract_slice with:
///   %slice = tensor.extract_slice %source[%o0, %o1][%s0, %s1][%t0, %t1]
static LogicalResult
matchRank2TensorExtractSlice(Value v, Value &source,
                             SmallVectorImpl<OpFoldResult> &offsets,
                             SmallVectorImpl<OpFoldResult> &sizes,
                             SmallVectorImpl<OpFoldResult> &strides) {
  auto extract =
      stripTensorCastWrappers(v).getDefiningOp<tensor::ExtractSliceOp>();
  if (!extract)
    return failure();

  auto resultTy = dyn_cast<RankedTensorType>(extract.getResult().getType());
  if (!resultTy || resultTy.getRank() != 2)
    return failure();

  auto mixedOffsets = extract.getMixedOffsets();
  auto mixedSizes = extract.getMixedSizes();
  auto mixedStrides = extract.getMixedStrides();

  if (mixedOffsets.size() != 2 || mixedSizes.size() != 2 ||
      mixedStrides.size() != 2)
    return failure();

  source = extract.getSource();
  offsets.assign(mixedOffsets.begin(), mixedOffsets.end());
  sizes.assign(mixedSizes.begin(), mixedSizes.end());
  strides.assign(mixedStrides.begin(), mixedStrides.end());
  return success();
}

/// Materialize a rank-2 memref view from a tensor slice-like value.
/// Supported sources:
///   - shard.all_slice over bufferization.to_tensor(memref)
///   - tensor.extract_slice over bufferization.to_tensor(memref)
static FailureOr<Value>
materializeInputAsSubviewRank2(OpBuilder &b, Location loc, Value inputTensor) {
  auto strippedOr = stripTensorCasts(inputTensor);
  if (failed(strippedOr))
    return failure();
  Value base = *strippedOr;

  Value sourceTensor;
  SmallVector<OpFoldResult> offsets, sizes, strides;

  if (auto allSlice = base.getDefiningOp<mlir::shard::AllSliceOp>()) {
    sourceTensor = allSlice->getOperand(0);

    auto resultTy = dyn_cast<RankedTensorType>(allSlice.getResult().getType());
    if (!resultTy || resultTy.getRank() != 2)
      return failure();

    auto func = allSlice->getParentOfType<func::FuncOp>();
    if (!func)
      return failure();

    auto linearIdxOrFail = [&]() -> FailureOr<Value> {
      auto args = func.getArguments();
      int64_t n = static_cast<int64_t>(args.size());
      if (n < 6)
        return failure();

      Value cid = args[n - 2];
      Value tid = args[n - 3];
      Value ntpc = args[n - 6];

      auto castToIndexIfNeeded = [&](Value v) -> Value {
        if (v.getType().isIndex())
          return v;
        if (isa<IntegerType>(v.getType()))
          return b.create<arith::IndexCastOp>(loc, b.getIndexType(), v);
        return Value();
      };

      cid = castToIndexIfNeeded(cid);
      tid = castToIndexIfNeeded(tid);
      ntpc = castToIndexIfNeeded(ntpc);
      if (!cid || !tid || !ntpc)
        return failure();

      Value mul = b.create<arith::MulIOp>(loc, cid, ntpc);
      Value add = b.create<arith::AddIOp>(loc, mul, tid);
      return add;
    }();

    if (failed(linearIdxOrFail))
      return failure();

    Value tileRows =
        b.create<arith::ConstantIndexOp>(loc, resultTy.getShape()[0]);
    Value off0 = b.create<arith::MulIOp>(loc, *linearIdxOrFail, tileRows);

    offsets = {off0, b.getIndexAttr(0)};
    sizes = {b.getIndexAttr(resultTy.getShape()[0]),
             b.getIndexAttr(resultTy.getShape()[1])};
    strides = {b.getIndexAttr(1), b.getIndexAttr(1)};
  } else if (auto extract = base.getDefiningOp<tensor::ExtractSliceOp>()) {
    sourceTensor = extract.getSource();

    auto resultTy = dyn_cast<RankedTensorType>(extract.getResult().getType());
    if (!resultTy || resultTy.getRank() != 2)
      return failure();

    auto mixedOffsets = extract.getMixedOffsets();
    auto mixedSizes = extract.getMixedSizes();
    auto mixedStrides = extract.getMixedStrides();
    if (mixedOffsets.size() != 2 || mixedSizes.size() != 2 ||
        mixedStrides.size() != 2)
      return failure();

    offsets.assign(mixedOffsets.begin(), mixedOffsets.end());
    sizes.assign(mixedSizes.begin(), mixedSizes.end());
    strides.assign(mixedStrides.begin(), mixedStrides.end());
  } else {
    return failure();
  }

  auto toTensor = sourceTensor.getDefiningOp<bufferization::ToTensorOp>();
  if (!toTensor)
    return failure();

  Value sourceMemref = toTensor->getOperand(0);
  auto sourceMemrefTy = dyn_cast<MemRefType>(sourceMemref.getType());
  if (!sourceMemrefTy || sourceMemrefTy.getRank() != 2)
    return failure();

  return buildRank2Subview(b, loc, sourceMemref, offsets, sizes, strides);
}

/// Build the memref local view corresponding to a tensor local slice.
/// Supported patterns:
///   - %t = shard.all_slice %src ...
///       where %src comes from bufferization.to_tensor %buf
///   - %t = tensor.extract_slice %src[%off][%size][1]
///       where %src comes from bufferization.to_tensor %buf
///
/// Returns a rank-1 memref.subview over the original memref source.
///
/// This helper creates any required arithmetic/subview IR at the builder's
/// current insertion point. Callers must ensure that point dominates the
/// future use sites.
static FailureOr<Value> materializeInputAsSubview(OpBuilder &b, Location loc,
                                                  Value inputTensor) {
  auto strippedOr = stripTensorCasts(inputTensor);
  if (failed(strippedOr))
    return failure();
  Value base = *strippedOr;

  Value sourceTensor;
  OpFoldResult offset;
  OpFoldResult size;
  OpFoldResult stride = b.getIndexAttr(1);

  if (auto allSlice = base.getDefiningOp<mlir::shard::AllSliceOp>()) {
    sourceTensor = allSlice->getOperand(0);

    auto resultTy = dyn_cast<RankedTensorType>(allSlice.getResult().getType());
    if (!resultTy || resultTy.getRank() != 1)
      return failure();

    int64_t sliceSize = resultTy.getShape()[0];
    if (sliceSize <= 0)
      return failure();

    auto func = allSlice->getParentOfType<func::FuncOp>();
    if (!func)
      return failure();

    auto linearIdxOrFail = [&]() -> FailureOr<Value> {
      // Same ABI convention used in ShardToLLVMPass, kept local here.
      auto args = func.getArguments();
      int64_t n = static_cast<int64_t>(args.size());
      if (n < 6)
        return failure();

      Value cid = args[n - 2];
      Value tid = args[n - 3];
      Value ntpc = args[n - 6];

      auto castToIndexIfNeeded = [&](Value v) -> Value {
        if (v.getType().isIndex())
          return v;
        if (isa<IntegerType>(v.getType()))
          return b.create<arith::IndexCastOp>(loc, b.getIndexType(), v);
        return Value();
      };

      cid = castToIndexIfNeeded(cid);
      tid = castToIndexIfNeeded(tid);
      ntpc = castToIndexIfNeeded(ntpc);
      if (!cid || !tid || !ntpc)
        return failure();

      Value mul = b.create<arith::MulIOp>(loc, cid, ntpc);
      Value add = b.create<arith::AddIOp>(loc, mul, tid);
      return add;
    }();

    if (failed(linearIdxOrFail))
      return failure();

    Value sliceSizeVal = b.create<arith::ConstantIndexOp>(loc, sliceSize);
    Value off = b.create<arith::MulIOp>(loc, *linearIdxOrFail, sliceSizeVal);

    offset = off;
    size = b.getIndexAttr(sliceSize);
  } else if (auto extract = base.getDefiningOp<tensor::ExtractSliceOp>()) {
    sourceTensor = extract.getSource();

    auto resultTy = dyn_cast<RankedTensorType>(extract.getResult().getType());
    if (!resultTy || resultTy.getRank() != 1)
      return failure();

    auto mixedOffsets = extract.getMixedOffsets();
    auto mixedSizes = extract.getMixedSizes();
    auto mixedStrides = extract.getMixedStrides();

    if (mixedOffsets.size() != 1 || mixedSizes.size() != 1 ||
        mixedStrides.size() != 1)
      return failure();

    offset = mixedOffsets[0];
    size = mixedSizes[0];
    stride = mixedStrides[0];
  } else {
    return failure();
  }

  auto toTensor = sourceTensor.getDefiningOp<bufferization::ToTensorOp>();
  if (!toTensor)
    return failure();

  Value sourceMemref = toTensor->getOperand(0);
  auto sourceMemrefTy = dyn_cast<MemRefType>(sourceMemref.getType());
  if (!sourceMemrefTy || sourceMemrefTy.getRank() != 1)
    return failure();

  int64_t staticSize = ShapedType::kDynamic;
  if (auto sizeAttr = dyn_cast<Attribute>(size)) {
    if (auto intAttr = dyn_cast<IntegerAttr>(sizeAttr))
      staticSize = intAttr.getInt();
  }

  auto subLayout = StridedLayoutAttr::get(sourceMemrefTy.getContext(),
                                          /*offset=*/ShapedType::kDynamic,
                                          /*strides=*/ArrayRef<int64_t>{1});

  auto subviewTy = MemRefType::get(ArrayRef<int64_t>{staticSize},
                                   sourceMemrefTy.getElementType(), subLayout,
                                   sourceMemrefTy.getMemorySpace());

  SmallVector<OpFoldResult> offsets = {offset};
  SmallVector<OpFoldResult> sizes = {size};
  SmallVector<OpFoldResult> strides = {stride};

  return b
      .create<memref::SubViewOp>(loc, subviewTy, sourceMemref, offsets, sizes,
                                 strides)
      .getResult();
}

/// Create a memref-semantics replacement for `localizedGeneric` at the
/// builder's current insertion point.
///
/// This helper intentionally does not change the insertion point. The caller
/// must position the builder such that all memref inputs and outputs dominate
/// the new generic.
static FailureOr<linalg::GenericOp> createMemrefGenericAtCurrentInsertionPoint(
    OpBuilder &b, linalg::GenericOp localizedGeneric, ValueRange memrefInputs,
    Value destSubview) {
  if (!isSimpleLocalizedElementwiseGeneric(localizedGeneric))
    return failure();

  auto destSubviewTy = dyn_cast<MemRefType>(destSubview.getType());
  if (!destSubviewTy || destSubviewTy.getRank() != 1)
    return failure();

  auto resultTy =
      dyn_cast<RankedTensorType>(localizedGeneric.getResult(0).getType());
  if (!resultTy || resultTy.getRank() != 1)
    return failure();

  if (destSubviewTy.getElementType() != resultTy.getElementType())
    return failure();

  if (static_cast<int64_t>(memrefInputs.size()) !=
      localizedGeneric.getNumDpsInputs())
    return failure();

  auto memrefGeneric = b.create<linalg::GenericOp>(
      localizedGeneric.getLoc(),
      /*resultTensorTypes=*/TypeRange{},
      /*inputs=*/memrefInputs,
      /*outputs=*/ValueRange{destSubview},
      /*indexingMaps=*/localizedGeneric.getIndexingMaps(),
      /*iteratorTypes=*/localizedGeneric.getIteratorTypes(),
      /*doc=*/nullptr,
      /*libraryCall=*/nullptr);

  Region &dstRegion = memrefGeneric.getRegion();
  dstRegion.getBlocks().clear();

  Block &srcBlock = localizedGeneric.getRegion().front();
  Block *dstBlock = new Block();
  dstRegion.push_back(dstBlock);

  for (BlockArgument a : srcBlock.getArguments())
    dstBlock->addArgument(a.getType(), a.getLoc());

  IRMapping map;
  for (auto [srcArg, dstArg] :
       llvm::zip(srcBlock.getArguments(), dstBlock->getArguments())) {
    map.map(srcArg, dstArg);
  }

  OpBuilder nb = OpBuilder::atBlockEnd(dstBlock);
  for (Operation &op : srcBlock.getOperations())
    nb.clone(op, map);

  return memrefGeneric;
}

static void eraseIfDead(Operation *op) {
  if (op && op->use_empty())
    op->erase();
}

static void removeNSPLocalizedAttrs(linalg::GenericOp generic) {
  if (!generic)
    return;
  generic->removeAttr("nsp.localized");
  generic->removeAttr("nsp.materialize_tile_shape");
  generic->removeAttr("nsp.materialize_split_axis");
  generic->removeAttr("nsp.group_id");
}

/// Return the source value after stripping trivial tensor.cast wrappers.
static Value stripTensorCastWrappers(Value v) {
  Value cur = v;
  while (auto castOp = cur.getDefiningOp<tensor::CastOp>())
    cur = castOp.getSource();
  return cur;
}

/// Return true iff `v` is the induction variable of `forOp`.
static bool isLoopIV(Value v, scf::ForOp forOp) {
  return v == forOp.getInductionVar();
}

/// Match a rank-1 tensor.extract_slice with:
///   %slice = tensor.extract_slice %source[%offset][%size][%stride]
///
/// On success, return the source and the single offset/size/stride triplet.
static LogicalResult matchRank1TensorExtractSlice(Value v, Value &source,
                                                  OpFoldResult &offset,
                                                  OpFoldResult &size,
                                                  OpFoldResult &stride) {
  auto extract =
      stripTensorCastWrappers(v).getDefiningOp<tensor::ExtractSliceOp>();
  if (!extract)
    return failure();

  auto resultTy = dyn_cast<RankedTensorType>(extract.getResult().getType());
  if (!resultTy || resultTy.getRank() != 1)
    return failure();

  auto mixedOffsets = extract.getMixedOffsets();
  auto mixedSizes = extract.getMixedSizes();
  auto mixedStrides = extract.getMixedStrides();

  if (mixedOffsets.size() != 1 || mixedSizes.size() != 1 ||
      mixedStrides.size() != 1)
    return failure();

  source = extract.getSource();
  offset = mixedOffsets[0];
  size = mixedSizes[0];
  stride = mixedStrides[0];
  return success();
}

/// Build a rank-1 memref.subview at the current insertion point.
static FailureOr<Value> buildRank1Subview(OpBuilder &b, Location loc,
                                          Value baseMemref, OpFoldResult offset,
                                          OpFoldResult size,
                                          OpFoldResult stride) {
  auto baseTy = dyn_cast<MemRefType>(baseMemref.getType());
  if (!baseTy || baseTy.getRank() != 1)
    return failure();

  int64_t staticSize = ShapedType::kDynamic;
  if (auto attr = dyn_cast<Attribute>(size))
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      staticSize = intAttr.getInt();

  auto subLayout = StridedLayoutAttr::get(baseTy.getContext(),
                                          /*offset=*/ShapedType::kDynamic,
                                          /*strides=*/ArrayRef<int64_t>{1});

  auto subviewTy =
      MemRefType::get(ArrayRef<int64_t>{staticSize}, baseTy.getElementType(),
                      subLayout, baseTy.getMemorySpace());

  SmallVector<OpFoldResult> offsets = {offset};
  SmallVector<OpFoldResult> sizes = {size};
  SmallVector<OpFoldResult> strides = {stride};

  return b
      .create<memref::SubViewOp>(loc, subviewTy, baseMemref, offsets, sizes,
                                 strides)
      .getResult();
}

/// Materialize a rank-1 memref view from a tensor slice-like value.
/// Supported sources:
///   - shard.all_slice over bufferization.to_tensor(memref)
///   - tensor.extract_slice over bufferization.to_tensor(memref)
static FailureOr<Value>
materializeTensorSliceLikeAsSubview(OpBuilder &b, Location loc, Value v) {
  return materializeInputAsSubview(b, loc, v);
}

/// Try to rewrite a tiled tensor scf.for into a memref-writing scf.for.
///
/// Expected pattern:
///   %init = tensor.empty
///   %res = scf.for ... iter_args(%acc = %init) -> tensor<tile x T> {
///     %in0 = tensor.extract_slice %input0[%iv] [chunk] [1]
///     %in1 = tensor.extract_slice %input1[%iv] [chunk] [1]   (optional)
///     %out = tensor.extract_slice %acc[%iv] [chunk] [1]
///     %v0 = vector.transfer_read %in0[%c0], ...
///     %v1 = vector.transfer_read %in1[%c0], ...              (optional)
///     ...
///     %tw = vector.transfer_write %vec, %out[%c0], ...
///     %next = tensor.insert_slice %tw into %acc[%iv] [chunk] [1]
///     scf.yield %next
///   }
///
/// Rewritten form:
///   %destSubview = ...
///   %inputSubview0 = ...
///   %inputSubview1 = ...                                     (optional)
///   scf.for ... {
///     %in0 = memref.subview %inputSubview0[%iv] [chunk] [1]
///     %in1 = memref.subview %inputSubview1[%iv] [chunk] [1] (optional)
///     %out = memref.subview %destSubview[%iv] [chunk] [1]
///     ...
///     vector.transfer_write %vec, %out[%c0], ...
///   }
static LogicalResult tryRewriteScfForTileToMemref(OpBuilder &b, Location loc,
                                                  scf::ForOp forOp,
                                                  Value destSubview) {
  if (forOp.getNumResults() != 1)
    return failure();
  if (forOp.getNumRegionIterArgs() != 1)
    return failure();

  auto resultTy = dyn_cast<RankedTensorType>(forOp.getResult(0).getType());
  if (!resultTy || resultTy.getRank() != 1)
    return failure();

  Value initArg = forOp.getInitArgs()[0];
  if (!initArg.getDefiningOp<tensor::EmptyOp>())
    return failure();

  Block *body = forOp.getBody();
  if (!body)
    return failure();

  auto yield = dyn_cast<scf::YieldOp>(body->getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return failure();

  auto insertSlice = yield.getOperand(0).getDefiningOp<tensor::InsertSliceOp>();
  if (!insertSlice)
    return failure();

  if (insertSlice.getDest() != body->getArgument(1))
    return failure();

  auto insertedTy =
      dyn_cast<RankedTensorType>(insertSlice.getSource().getType());
  if (!insertedTy || insertedTy.getRank() != 1)
    return failure();

  auto insertOffsets = insertSlice.getMixedOffsets();
  auto insertSizes = insertSlice.getMixedSizes();
  auto insertStrides = insertSlice.getMixedStrides();
  if (insertOffsets.size() != 1 || insertSizes.size() != 1 ||
      insertStrides.size() != 1)
    return failure();

  // Require the inserted slice offset to be the loop IV.
  if (auto offVal = dyn_cast<Value>(insertOffsets[0])) {
    if (!isLoopIV(offVal, forOp))
      return failure();
  } else {
    return failure();
  }

  // The inserted source is typically the result of vector.transfer_write.
  auto transferWrite =
      insertSlice.getSource().getDefiningOp<vector::TransferWriteOp>();
  if (!transferWrite)
    return failure();

  // The destination of transfer_write must be the chunk extracted from the loop
  // carried tensor.
  Value accChunkSource;
  OpFoldResult accChunkOffset, accChunkSize, accChunkStride;
  if (failed(matchRank1TensorExtractSlice(transferWrite.getBase(),
                                          accChunkSource, accChunkOffset,
                                          accChunkSize, accChunkStride)))
    return failure();

  if (accChunkSource != body->getArgument(1))
    return failure();

  if (auto offVal = dyn_cast<Value>(accChunkOffset)) {
    if (!isLoopIV(offVal, forOp))
      return failure();
  } else {
    return failure();
  }

  // Collect transfer reads from the body.
  SmallVector<vector::TransferReadOp> transferReads;
  for (Operation &op : body->without_terminator()) {
    if (auto tr = dyn_cast<vector::TransferReadOp>(op))
      transferReads.push_back(tr);
  }

  if (transferReads.empty() || transferReads.size() > 2)
    return failure();

  // Each transfer_read source must be an extract_slice with IV offset.
  SmallVector<Value> inputBaseTensors;
  SmallVector<OpFoldResult> chunkOffsets;
  SmallVector<OpFoldResult> chunkSizes;
  SmallVector<OpFoldResult> chunkStrides;
  inputBaseTensors.reserve(transferReads.size());

  for (vector::TransferReadOp tr : transferReads) {
    Value srcTensor;
    OpFoldResult off, size, stride;
    if (failed(matchRank1TensorExtractSlice(tr.getBase(), srcTensor, off, size,
                                            stride)))
      return failure();

    if (auto offVal = dyn_cast<Value>(off)) {
      if (!isLoopIV(offVal, forOp))
        return failure();
    } else {
      return failure();
    }

    inputBaseTensors.push_back(srcTensor);
    chunkOffsets.push_back(off);
    chunkSizes.push_back(size);
    chunkStrides.push_back(stride);
  }

  // Rebuild the outer input tiles as memrefs.
  SmallVector<Value> inputTileMemrefs;
  inputTileMemrefs.reserve(inputBaseTensors.size());
  for (Value tensorLike : inputBaseTensors) {
    auto memrefOr = materializeTensorSliceLikeAsSubview(b, loc, tensorLike);
    if (failed(memrefOr))
      return failure();
    inputTileMemrefs.push_back(*memrefOr);
  }

  // Create the replacement loop at the current insertion point.
  auto newFor = b.create<scf::ForOp>(loc, forOp.getLowerBound(),
                                     forOp.getUpperBound(), forOp.getStep());

  Block *newBody = newFor.getBody();
  newBody->getOperations().clear();

  IRMapping map;
  map.map(forOp.getInductionVar(), newFor.getInductionVar());

  OpBuilder nb = OpBuilder::atBlockEnd(newBody);

  // Materialize input/output chunk memrefs inside the loop.
  SmallVector<Value> newChunkMemrefs;
  newChunkMemrefs.reserve(inputTileMemrefs.size());
  for (size_t i = 0; i < inputTileMemrefs.size(); ++i) {
    auto chunkOr = buildRank1Subview(
        nb, loc, inputTileMemrefs[i],
        map.lookupOrDefault(dyn_cast_if_present<Value>(chunkOffsets[i])),
        chunkSizes[i], chunkStrides[i]);
    if (failed(chunkOr))
      return failure();
    newChunkMemrefs.push_back(*chunkOr);
  }

  auto outChunkOr =
      buildRank1Subview(nb, loc, destSubview, newFor.getInductionVar(),
                        accChunkSize, accChunkStride);
  if (failed(outChunkOr))
    return failure();
  Value outChunkMemref = *outChunkOr;

  // First clone every op except tensor.extract_slice / tensor.insert_slice /
  // scf.yield. Replace transfer_{read,write} sources/dests as needed.
  for (Operation &op : body->without_terminator()) {
    if (isa<tensor::ExtractSliceOp>(op) || isa<tensor::InsertSliceOp>(op))
      continue;

    if (auto tr = dyn_cast<vector::TransferReadOp>(op)) {
      size_t idx = 0;
      bool matched = false;
      for (; idx < transferReads.size(); ++idx) {
        if (transferReads[idx] == tr) {
          matched = true;
          break;
        }
      }
      if (!matched)
        return failure();

      SmallVector<Value> indices;
      indices.reserve(tr.getIndices().size());
      for (Value iv : tr.getIndices())
        indices.push_back(map.lookupOrDefault(iv));

      auto newRead = nb.create<vector::TransferReadOp>(
          tr.getLoc(), tr.getVectorType(), newChunkMemrefs[idx], indices,
          tr.getPermutationMapAttr(), tr.getPadding(), tr.getMask(),
          tr.getInBoundsAttr());
      map.map(tr.getResult(), newRead.getResult());
      continue;
    }

    if (auto tw = dyn_cast<vector::TransferWriteOp>(op)) {
      SmallVector<Value> indices;
      indices.reserve(tw.getIndices().size());
      for (Value iv : tw.getIndices())
        indices.push_back(map.lookupOrDefault(iv));

      nb.create<vector::TransferWriteOp>(
          tw.getLoc(), map.lookupOrDefault(tw.getVector()), outChunkMemref,
          indices, tw.getPermutationMapAttr(), tw.getMask(),
          tw.getInBoundsAttr());
      continue;
    }

    Operation *cloned = nb.clone(op, map);
    for (auto it : llvm::zip(op.getResults(), cloned->getResults()))
      map.map(std::get<0>(it), std::get<1>(it));
  }

  nb.create<scf::YieldOp>(loc);
  return success();
}

/// Try to rewrite the rank-2 tiled tensor pattern produced by vectorization
/// into nested memref-writing scf.for loops.
///
/// Expected shape (current fast-path):
///   %init = tensor.empty : tensor<T0xT1xf32>
///   %outer = scf.for %i = ... iter_args(%acc0 = %init) -> tensor<T0xT1xf32> {
///     %inner = scf.for %j = ... iter_args(%acc1 = %acc0) -> tensor<T0xT1xf32>
///     {
///       %a = tensor.extract_slice %in0[%i, %j] [1, C] [1, 1]
///       %b = tensor.extract_slice %in1[%i, %j] [1, C] [1, 1]   (optional)
///       %o = tensor.extract_slice %acc1[%i, %j] [1, C] [1, 1]
///       %ra = vector.transfer_read %a[%c0, %c0], ...
///       %rb = vector.transfer_read %b[%c0, %c0], ...
///       ...
///       %tw = vector.transfer_write %vec, %o[%c0, %c0], ...
///       %next = tensor.insert_slice %tw into %acc1[%i, %j] [1, C] [1, 1]
///       scf.yield %next
///     }
///     scf.yield %inner
///   }
///
/// Rewritten form:
///   %in0Tile = memref.subview ...
///   %in1Tile = memref.subview ...
///   scf.for %i = ... {
///     scf.for %j = ... {
///       %a = memref.subview %in0Tile[%i, %j] [1, C] [1, 1]
///       %b = memref.subview %in1Tile[%i, %j] [1, C] [1, 1]
///       %o = memref.subview %destSubview[%i, %j] [1, C] [1, 1]
///       ...
///       vector.transfer_write %vec, %o[%c0, %c0], ...
///     }
///   }
static LogicalResult tryRewriteRank2ScfForTileToMemref(OpBuilder &b,
                                                       Location loc,
                                                       scf::ForOp outerFor,
                                                       Value destSubview) {
  if (outerFor.getNumResults() != 1 || outerFor.getNumRegionIterArgs() != 1)
    return failure();

  auto outerResultTy =
      dyn_cast<RankedTensorType>(outerFor.getResult(0).getType());
  if (!outerResultTy || outerResultTy.getRank() != 2)
    return failure();

  Value outerInit = outerFor.getInitArgs()[0];
  if (!outerInit.getDefiningOp<tensor::EmptyOp>())
    return failure();

  Block *outerBody = outerFor.getBody();
  if (!outerBody)
    return failure();

  auto outerYield = dyn_cast<scf::YieldOp>(outerBody->getTerminator());
  if (!outerYield || outerYield.getNumOperands() != 1)
    return failure();

  auto innerFor = outerYield.getOperand(0).getDefiningOp<scf::ForOp>();
  if (!innerFor)
    return failure();
  if (innerFor->getBlock() != outerBody)
    return failure();

  if (innerFor.getNumResults() != 1 || innerFor.getNumRegionIterArgs() != 1)
    return failure();

  auto innerResultTy =
      dyn_cast<RankedTensorType>(innerFor.getResult(0).getType());
  if (!innerResultTy || innerResultTy != outerResultTy)
    return failure();

  if (innerFor.getInitArgs()[0] != outerBody->getArgument(1))
    return failure();

  Block *innerBody = innerFor.getBody();
  if (!innerBody)
    return failure();

  auto innerYield = dyn_cast<scf::YieldOp>(innerBody->getTerminator());
  if (!innerYield || innerYield.getNumOperands() != 1)
    return failure();

  auto insertSlice =
      innerYield.getOperand(0).getDefiningOp<tensor::InsertSliceOp>();
  if (!insertSlice)
    return failure();
  if (insertSlice.getDest() != innerBody->getArgument(1))
    return failure();

  SmallVector<OpFoldResult> insertOffsets, insertSizes, insertStrides;
  insertOffsets.assign(insertSlice.getMixedOffsets().begin(),
                       insertSlice.getMixedOffsets().end());
  insertSizes.assign(insertSlice.getMixedSizes().begin(),
                     insertSlice.getMixedSizes().end());
  insertStrides.assign(insertSlice.getMixedStrides().begin(),
                       insertSlice.getMixedStrides().end());

  if (insertOffsets.size() != 2 || insertSizes.size() != 2 ||
      insertStrides.size() != 2)
    return failure();

  auto rowOff = dyn_cast<Value>(insertOffsets[0]);
  auto colOff = dyn_cast<Value>(insertOffsets[1]);
  if (!rowOff || !colOff)
    return failure();
  if (!isLoopIV(rowOff, outerFor) || !isLoopIV(colOff, innerFor))
    return failure();

  auto transferWrite =
      insertSlice.getSource().getDefiningOp<vector::TransferWriteOp>();
  if (!transferWrite)
    return failure();

  Value accChunkSource;
  SmallVector<OpFoldResult> accChunkOffsets, accChunkSizes, accChunkStrides;
  if (failed(matchRank2TensorExtractSlice(transferWrite.getBase(),
                                          accChunkSource, accChunkOffsets,
                                          accChunkSizes, accChunkStrides)))
    return failure();

  if (accChunkSource != innerBody->getArgument(1))
    return failure();

  auto accRowOff = dyn_cast<Value>(accChunkOffsets[0]);
  auto accColOff = dyn_cast<Value>(accChunkOffsets[1]);
  if (!accRowOff || !accColOff)
    return failure();
  if (!isLoopIV(accRowOff, outerFor) || !isLoopIV(accColOff, innerFor))
    return failure();

  SmallVector<vector::TransferReadOp> transferReads;
  for (Operation &op : innerBody->without_terminator()) {
    if (auto tr = dyn_cast<vector::TransferReadOp>(op))
      transferReads.push_back(tr);
  }
  if (transferReads.empty() || transferReads.size() > 2)
    return failure();

  SmallVector<Value> inputBaseTensors;
  SmallVector<SmallVector<OpFoldResult>> inputOffsets, inputSizes, inputStrides;
  inputBaseTensors.reserve(transferReads.size());

  for (vector::TransferReadOp tr : transferReads) {
    Value srcTensor;
    SmallVector<OpFoldResult> offs, szs, strs;
    if (failed(matchRank2TensorExtractSlice(tr.getBase(), srcTensor, offs, szs,
                                            strs)))
      return failure();

    auto inRowOff = dyn_cast<Value>(offs[0]);
    auto inColOff = dyn_cast<Value>(offs[1]);
    if (!inRowOff || !inColOff)
      return failure();
    if (!isLoopIV(inRowOff, outerFor) || !isLoopIV(inColOff, innerFor))
      return failure();

    inputBaseTensors.push_back(srcTensor);
    inputOffsets.push_back(std::move(offs));
    inputSizes.push_back(std::move(szs));
    inputStrides.push_back(std::move(strs));
  }

  SmallVector<Value> inputTileMemrefs;
  inputTileMemrefs.reserve(inputBaseTensors.size());
  for (Value tensorLike : inputBaseTensors) {
    auto memrefOr = materializeInputAsSubviewRank2(b, loc, tensorLike);
    if (failed(memrefOr))
      return failure();
    inputTileMemrefs.push_back(*memrefOr);
  }

  auto newOuter =
      b.create<scf::ForOp>(loc, outerFor.getLowerBound(),
                           outerFor.getUpperBound(), outerFor.getStep());
  Block *newOuterBody = newOuter.getBody();
  newOuterBody->getOperations().clear();

  IRMapping outerMap;
  outerMap.map(outerFor.getInductionVar(), newOuter.getInductionVar());
  OpBuilder ob = OpBuilder::atBlockEnd(newOuterBody);

  auto newInner =
      ob.create<scf::ForOp>(loc, innerFor.getLowerBound(),
                            innerFor.getUpperBound(), innerFor.getStep());
  Block *newInnerBody = newInner.getBody();
  newInnerBody->getOperations().clear();

  IRMapping map = outerMap;
  map.map(innerFor.getInductionVar(), newInner.getInductionVar());

  OpBuilder ib = OpBuilder::atBlockEnd(newInnerBody);

  SmallVector<Value> newChunkMemrefs;
  newChunkMemrefs.reserve(inputTileMemrefs.size());
  for (size_t i = 0; i < inputTileMemrefs.size(); ++i) {
    SmallVector<OpFoldResult> offs = {
        map.lookupOrDefault(dyn_cast_if_present<Value>(inputOffsets[i][0])),
        map.lookupOrDefault(dyn_cast_if_present<Value>(inputOffsets[i][1]))};
    auto chunkOr = buildRank2Subview(ib, loc, inputTileMemrefs[i], offs,
                                     inputSizes[i], inputStrides[i]);
    if (failed(chunkOr))
      return failure();
    newChunkMemrefs.push_back(*chunkOr);
  }

  SmallVector<OpFoldResult> outOffsets = {newOuter.getInductionVar(),
                                          newInner.getInductionVar()};
  auto outChunkOr = buildRank2Subview(ib, loc, destSubview, outOffsets,
                                      accChunkSizes, accChunkStrides);
  if (failed(outChunkOr))
    return failure();
  Value outChunkMemref = *outChunkOr;

  for (Operation &op : innerBody->without_terminator()) {
    if (isa<tensor::ExtractSliceOp>(op) || isa<tensor::InsertSliceOp>(op))
      continue;

    if (auto tr = dyn_cast<vector::TransferReadOp>(op)) {
      size_t idx = 0;
      bool matched = false;
      for (; idx < transferReads.size(); ++idx) {
        if (transferReads[idx] == tr) {
          matched = true;
          break;
        }
      }
      if (!matched)
        return failure();

      SmallVector<Value> indices;
      indices.reserve(tr.getIndices().size());
      for (Value iv : tr.getIndices())
        indices.push_back(map.lookupOrDefault(iv));

      auto newRead = ib.create<vector::TransferReadOp>(
          tr.getLoc(), tr.getVectorType(), newChunkMemrefs[idx], indices,
          tr.getPermutationMapAttr(), tr.getPadding(), tr.getMask(),
          tr.getInBoundsAttr());
      map.map(tr.getResult(), newRead.getResult());
      continue;
    }

    if (auto tw = dyn_cast<vector::TransferWriteOp>(op)) {
      SmallVector<Value> indices;
      indices.reserve(tw.getIndices().size());
      for (Value iv : tw.getIndices())
        indices.push_back(map.lookupOrDefault(iv));

      ib.create<vector::TransferWriteOp>(
          tw.getLoc(), map.lookupOrDefault(tw.getVector()), outChunkMemref,
          indices, tw.getPermutationMapAttr(), tw.getMask(),
          tw.getInBoundsAttr());
      continue;
    }

    Operation *cloned = ib.clone(op, map);
    for (auto it : llvm::zip(op.getResults(), cloned->getResults()))
      map.map(std::get<0>(it), std::get<1>(it));
  }

  ib.create<scf::YieldOp>(loc);
  ob.create<scf::YieldOp>(loc);
  return success();
}

static LogicalResult
materializeTileToDestination(OpBuilder &b,
                             mlir::hexagon::nsp::MaterializeTileOp tileOp,
                             mlir::shard::GridOp grid) {
  Location loc = tileOp.getLoc();

  Value source = tileOp.getSource();
  Value dest = tileOp.getDest();

  auto sourceTy = dyn_cast<RankedTensorType>(source.getType());
  auto destTy = dyn_cast<MemRefType>(dest.getType());
  if (!sourceTy || !destTy)
    return failure();

  if (sourceTy.getRank() != destTy.getRank())
    return failure();

  int64_t rank = sourceTy.getRank();
  if (rank != 1 && rank != 2)
    return failure();

  int64_t splitAxis = tileOp.getSplitAxis();
  if (splitAxis != 0)
    return failure();

  auto tileShapeOr = getStaticTileShape(tileOp);
  if (failed(tileShapeOr))
    return failure();
  SmallVector<int64_t> tileShape = *tileShapeOr;

  if ((int64_t)tileShape.size() != rank)
    return failure();

  // All Pattern's path IR must be created at the tile op insertion point
  // so that the generated values dominate the new memref generic.
  b.setInsertionPoint(tileOp);
  Value procIdx = b.create<mlir::shard::ProcessLinearIndexOp>(loc, grid);
  Value tileExtent0 = b.create<arith::ConstantIndexOp>(loc, tileShape[0]);
  Value offset0 = b.create<arith::MulIOp>(loc, procIdx, tileExtent0);

  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;

  offsets.reserve(rank);
  sizes.reserve(rank);
  strides.reserve(rank);

  for (int64_t d = 0; d < rank; ++d) {
    offsets.push_back(d == splitAxis ? OpFoldResult(offset0)
                                     : OpFoldResult(b.getIndexAttr(0)));
    sizes.push_back(b.getIndexAttr(tileShape[d]));
    strides.push_back(b.getIndexAttr(1));
  }

  auto subview =
      b.create<memref::SubViewOp>(loc, dest, offsets, sizes, strides);
  Value destSubview = subview.getResult();

  // Pattern 1: localized elementwise linalg.generic.
  if (auto localizedGeneric = source.getDefiningOp<linalg::GenericOp>()) {
    Value oldInit = localizedGeneric.getDpsInitOperand(0)->get();
    Operation *oldGenericOp = localizedGeneric.getOperation();

    if (isSimpleLocalizedElementwiseGeneric(localizedGeneric)) {
      SmallVector<Value> memrefInputs;
      memrefInputs.reserve(localizedGeneric.getNumDpsInputs());

      bool canRewriteAllInputs = true;
      for (OpOperand *in : localizedGeneric.getDpsInputOperands()) {
        auto subviewOr = materializeInputAsSubview(b, loc, in->get());
        if (failed(subviewOr)) {
          canRewriteAllInputs = false;
          break;
        }
        memrefInputs.push_back(*subviewOr);
      }

      if (canRewriteAllInputs) {
        auto newGenericOr = createMemrefGenericAtCurrentInsertionPoint(
            b, localizedGeneric, memrefInputs, destSubview);
        if (succeeded(newGenericOr)) {
          removeNSPLocalizedAttrs(localizedGeneric);

          // Erase the hand-off op first, since it is the last user of the
          // old tensor result.
          tileOp.erase();

          if (oldGenericOp->use_empty())
            oldGenericOp->erase();

          if (auto emptyOp = oldInit.getDefiningOp<tensor::EmptyOp>())
            eraseIfDead(emptyOp);

          return success();
        }
      }
    }
  }

  // Pattern 2: tiled tensor scf.for building the local tile by insert_slice.
  if (auto tileLoop = source.getDefiningOp<scf::ForOp>()) {
    Value initArg = tileLoop.getInitArgs().front();
    Operation *oldLoopOp = tileLoop.getOperation();

    auto sourceTy = dyn_cast<RankedTensorType>(source.getType());
    if (sourceTy && sourceTy.getRank() == 2) {
      if (succeeded(tryRewriteRank2ScfForTileToMemref(b, loc, tileLoop,
                                                      destSubview))) {
        tileOp.erase();

        if (oldLoopOp->use_empty())
          oldLoopOp->erase();

        if (auto emptyOp = initArg.getDefiningOp<tensor::EmptyOp>())
          eraseIfDead(emptyOp);

        return success();
      }
    }

    // Keep the existing rank-1 fast-path unchanged.
    // If rank-2 matching fails, we still fall back to the old logic below.
    if (succeeded(
            tryRewriteScfForTileToMemref(b, loc, tileLoop, destSubview))) {
      tileOp.erase();

      if (oldLoopOp->use_empty())
        oldLoopOp->erase();

      if (auto emptyOp = initArg.getDefiningOp<tensor::EmptyOp>())
        eraseIfDead(emptyOp);

      return success();
    }
  }

  // Fallback: keep the original tensor path.
  b.create<bufferization::MaterializeInDestinationOp>(
      loc, TypeRange{}, source, destSubview,
      /*restrict=*/false, /*writable=*/true);

  if (auto localizedGeneric = source.getDefiningOp<linalg::GenericOp>())
    removeNSPLocalizedAttrs(localizedGeneric);

  tileOp.erase();
  return success();
}

struct NSPMaterializePass
    : public PassWrapper<NSPMaterializePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NSPMaterializePass)

  StringRef getArgument() const final { return "nsp-materialize"; }

  StringRef getDescription() const final {
    return "Materialize shard-local tensor results into destination memrefs";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::shard::ShardDialect, mlir::arith::ArithDialect,
                    mlir::bufferization::BufferizationDialect,
                    mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                    mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                    mlir::tensor::TensorDialect, mlir::vector::VectorDialect,
                    mlir::hexagon::nsp::NSPDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();

    SmallVector<mlir::hexagon::nsp::MaterializeTileOp> worklist;
    module.walk([&](mlir::hexagon::nsp::MaterializeTileOp op) {
      worklist.push_back(op);
    });

    for (mlir::hexagon::nsp::MaterializeTileOp tileOp : worklist) {
      if (!tileOp || !tileOp->getParentOp())
        continue;

      auto gridOr = lookupGrid(tileOp, tileOp.getGridAttr());
      if (failed(gridOr)) {
        tileOp.emitError() << "NSPMaterialize: could not resolve grid symbol "
                           << tileOp.getGridAttr();
        signalPassFailure();
        return;
      }

      OpBuilder b(tileOp);
      if (failed(materializeTileToDestination(b, tileOp, *gridOr))) {
        tileOp.emitError()
            << "NSPMaterialize: failed to materialize local tile into "
               "destination";
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createNSPMaterializePass() {
  return std::make_unique<NSPMaterializePass>();
}

} // namespace hexagon
} // namespace mlir
