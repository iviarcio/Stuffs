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
//   3. tries fast-path rewrites that replace tensor tile materialization with
//      direct writes into memref subviews,
//   4. falls back to bufferization.materialize_in_destination when no
//      supported fast-path matches,
//   5. erases the temporary NSP hand-off op.
//
// Bring-up constraints:
//   - the generic tile hand-off supports rank-1 / rank-2 destination
//     materialization through tile_shape,
//   - split_axis is currently restricted to 0,
//   - the direct memref rewrite for localized linalg.generic remains rank-1,
//   - a dedicated fast-path exists for the elemenwise 2d-style pattern:
//       * rank-2 outer tile / accumulator
//       * rank-1 inner vectorized chunks
//   - other rank-2 patterns still fall back to
//     bufferization.materialize_in_destination,
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

#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "hexagon/Dialect/NSP/IR/NSPDialect.h"
#include "hexagon/Dialect/NSP/IR/NSPOps.h"

namespace mlir {
namespace hexagon {

namespace {

//===----------------------------------------------------------------------===//
// Grid / ABI helpers
//===----------------------------------------------------------------------===//

static Value castToIndexIfNeeded(Value v, OpBuilder &b, Location loc) {
  if (v.getType().isIndex())
    return v;
  if (isa<IntegerType>(v.getType()))
    return b.create<arith::IndexCastOp>(loc, b.getIndexType(), v);
  return Value();
}

/// Compute the participant linear index from the Hexagon entry-point ABI:
///   linearIdx = coreId * numThreadsPerCore + threadId
static FailureOr<Value>
computeLinearIdxFromFuncArgs(func::FuncOp func, OpBuilder &b, Location loc) {
  auto args = func.getArguments();
  int64_t n = static_cast<int64_t>(args.size());
  if (n < 6)
    return failure();

  Value cid = castToIndexIfNeeded(args[n - 2], b, loc);
  Value tid = castToIndexIfNeeded(args[n - 3], b, loc);
  Value ntpc = castToIndexIfNeeded(args[n - 6], b, loc);
  if (!cid || !tid || !ntpc)
    return failure();

  Value mul = b.create<arith::MulIOp>(loc, cid, ntpc);
  return b.create<arith::AddIOp>(loc, mul, tid).getResult();
}

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

//===----------------------------------------------------------------------===//
// Generic pattern classification helpers
//===----------------------------------------------------------------------===//

/// Return the source value after stripping trivial tensor.cast wrappers.
static Value stripTensorCastWrappers(Value v) {
  Value cur = v;
  while (auto castOp = cur.getDefiningOp<tensor::CastOp>())
    cur = castOp.getSource();
  return cur;
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

/// Clone an elementwise-like op whose operands are already mapped,
/// preserving all attributes. Do not clone ops with regions or side
/// efects.
static FailureOr<Operation *> cloneMappedPureOp(OpBuilder &b, Operation &op,
                                                IRMapping &map) {
  if (op.getNumRegions() != 0)
    return failure();

  // StringRef dialect = op.getName().getDialectNamespace();
  // if (dialect != "arith" && dialect != "math")
  //   return failure();
  if (!isMemoryEffectFree(&op))
    return failure();

  SmallVector<Value> operands;
  operands.reserve(op.getNumOperands());

  for (Value operand : op.getOperands()) {
    Value mapped = map.lookupOrNull(operand);
    if (!mapped)
      return failure();
    operands.push_back(mapped);
  }

  SmallVector<Type> resultTypes(op.getResultTypes());
  OperationState state(op.getLoc(), op.getName());
  state.addOperands(operands);
  state.addTypes(resultTypes);
  state.addAttributes(op.getAttrs());

  Operation *newOp = b.create(state);

  for (auto [oldResult, newResult] :
       llvm::zip(op.getResults(), newOp->getResults()))
    map.map(oldResult, newResult);

  return newOp;
}

//===----------------------------------------------------------------------===//
// Memref view builders
//===----------------------------------------------------------------------===//

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

  auto subview =
      b.create<memref::SubViewOp>(loc, baseMemref, offsets, sizes, strides);
  return subview.getResult();
}

/// Build a rank-1 memref view for a [1 x C] slice taken from a rank-2 base
/// memref at offsets [rowOff, colOff].
///
/// This helper is used by the current 2-D elementwise vectorized fast-path,
/// where:
///   - the outer tile / accumulator is rank-2, but
///   - the vector.transfer_{read,write} ops operate on rank-1 chunks.
///
/// In the source tensor IR this corresponds to patterns such as:
///   %chunk = tensor.extract_slice %tile[%i, %j] [1, C] [1, 1]
///            : tensor<T0xT1xf32> to tensor<Cxf32>
///
/// The memref form is built in two steps:
///   1. create a rank-2 memref.subview of shape [1 x C]
///   2. collapse it to rank-1 so it matches vector.transfer_read/write [%c0]
static FailureOr<Value> buildRank1SubviewFromRank2(OpBuilder &b, Location loc,
                                                   Value baseMemref,
                                                   OpFoldResult rowOff,
                                                   OpFoldResult colOff,
                                                   OpFoldResult chunkSize) {
  auto baseTy = dyn_cast<MemRefType>(baseMemref.getType());
  if (!baseTy || baseTy.getRank() != 2)
    return failure();

  int64_t staticSize = ShapedType::kDynamic;
  if (auto attr = dyn_cast<Attribute>(chunkSize))
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      staticSize = intAttr.getInt();

  auto subLayout = StridedLayoutAttr::get(baseTy.getContext(),
                                          /*offset=*/ShapedType::kDynamic,
                                          /*strides=*/ArrayRef<int64_t>{1});

  auto subviewTy =
      MemRefType::get(ArrayRef<int64_t>{staticSize}, baseTy.getElementType(),
                      subLayout, baseTy.getMemorySpace());

  SmallVector<OpFoldResult> offsets = {rowOff, colOff};
  SmallVector<OpFoldResult> sizes = {b.getIndexAttr(1), chunkSize};
  SmallVector<OpFoldResult> strides = {b.getIndexAttr(1), b.getIndexAttr(1)};

  auto full2D =
      b.create<memref::SubViewOp>(loc, baseMemref, offsets, sizes, strides);

  return b
      .create<memref::CollapseShapeOp>(loc, subviewTy, full2D.getResult(),
                                       ReassociationIndices{{0, 1}})
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

//===----------------------------------------------------------------------===//
// Tensor -> memref materialization helpers
//===----------------------------------------------------------------------===//

/// Materialize a rank-2 memref view from a tensor slice-like value.
/// Supported sources:
///   - shard.all_slice over bufferization.to_tensor(memref)
///   - tensor.extract_slice over bufferization.to_tensor(memref)
static FailureOr<Value>
materializeRank2InputAsSubview(OpBuilder &b, Location loc, Value inputTensor) {
  Value base = stripTensorCastWrappers(inputTensor);

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

    auto linearIdxOrFail = computeLinearIdxFromFuncArgs(func, b, loc);
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

  Value baseTensor = sourceTensor;
  while (true) {
    if (auto castOp = baseTensor.getDefiningOp<tensor::CastOp>()) {
      baseTensor = castOp.getSource();
      continue;
    }
    if (auto allSliceOp = baseTensor.getDefiningOp<mlir::shard::AllSliceOp>()) {
      auto resultTy =
          dyn_cast<RankedTensorType>(allSliceOp.getResult().getType());
      if (!resultTy || resultTy.getRank() != 2)
        return failure();

      auto func = allSliceOp->getParentOfType<func::FuncOp>();
      if (!func)
        return failure();

      auto linearIdxOrFail = computeLinearIdxFromFuncArgs(func, b, loc);
      if (failed(linearIdxOrFail))
        return failure();

      Value tileRows =
          b.create<arith::ConstantIndexOp>(loc, resultTy.getShape()[0]);
      Value off0 = b.create<arith::MulIOp>(loc, *linearIdxOrFail, tileRows);

      // Compose the existing offsets with the all_slice outer tile offset.
      if (offsets.empty()) {
        offsets = {off0, b.getIndexAttr(0)};
        sizes = {b.getIndexAttr(resultTy.getShape()[0]),
                 b.getIndexAttr(resultTy.getShape()[1])};
        strides = {b.getIndexAttr(1), b.getIndexAttr(1)};
      } else {
        if (auto old0 = dyn_cast<Value>(offsets[0])) {
          Value newOff0 = b.create<arith::AddIOp>(loc, off0, old0).getResult();
          offsets[0] = newOff0;
        } else if (auto old0Attr = dyn_cast<Attribute>(offsets[0])) {
          Value old0Val = b.create<arith::ConstantIndexOp>(
                               loc, cast<IntegerAttr>(old0Attr).getInt())
                              .getResult();
          Value newOff0 =
              b.create<arith::AddIOp>(loc, off0, old0Val).getResult();
          offsets[0] = newOff0;
        } else {
          return failure();
        }
      }

      baseTensor = allSliceOp.getOperand();
      continue;
    }
    break;
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
static FailureOr<Value>
materializeRank1InputAsSubview(OpBuilder &b, Location loc, Value inputTensor) {
  Value base = stripTensorCastWrappers(inputTensor);

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

    auto linearIdxOrFail = computeLinearIdxFromFuncArgs(func, b, loc);

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

/// Materialize a scalar value from a rank-1 local tensor at `index`.
///
/// Supported sources:
///   - shard.all_slice / tensor.extract_slice over
///   bufferization.to_tensor(memref)
///   - simple localized rank-1 linalg.generic chains
///
/// This is used for row-wise broadcasted inputs that appear after localization
/// as:
///   tensor.extract_slice %rowVec[%i] [1] [1] : tensor<T> to tensor<f32>
static FailureOr<Value>
materializeScalarFromRank1TensorAtIndex(OpBuilder &b, Location loc,
                                        Value tensorLike, Value index) {
  Value base = stripTensorCastWrappers(tensorLike);

  // Case 1: directly backed by a rank-1 memref-ish source.
  if (succeeded(materializeRank1InputAsSubview(b, loc, base))) {
    auto memrefOr = materializeRank1InputAsSubview(b, loc, base);
    Value memref = *memrefOr;
    auto memrefTy = dyn_cast<MemRefType>(memref.getType());
    if (!memrefTy || memrefTy.getRank() != 1)
      return failure();
    return b.create<memref::LoadOp>(loc, memref, ValueRange{index}).getResult();
  }

  // Case 2: simple localized rank-1 generic.
  auto generic = base.getDefiningOp<linalg::GenericOp>();
  if (!generic || !isSimpleLocalizedElementwiseGeneric(generic))
    return failure();

  auto resultTy = dyn_cast<RankedTensorType>(generic.getResult(0).getType());
  if (!resultTy || resultTy.getRank() != 1)
    return failure();

  const int64_t numInputs = generic.getNumDpsInputs();
  SmallVector<Value> scalarInputs;
  scalarInputs.reserve(numInputs);

  for (OpOperand *in : generic.getDpsInputOperands()) {
    auto scalarOr =
        materializeScalarFromRank1TensorAtIndex(b, loc, in->get(), index);
    if (failed(scalarOr))
      return failure();
    scalarInputs.push_back(*scalarOr);
  }

  Block &srcBlock = generic.getRegion().front();
  if (static_cast<int64_t>(srcBlock.getNumArguments()) != numInputs + 1)
    return failure();

  // Keep bring-up conservative: only support bodies that do not depend on the
  // output argument.
  if (!srcBlock.getArgument(numInputs).use_empty())
    return failure();

  IRMapping map;
  for (int64_t i = 0; i < numInputs; ++i)
    map.map(srcBlock.getArgument(i), scalarInputs[i]);

  OpBuilder nb = b;
  for (Operation &op : srcBlock.without_terminator()) {
    auto clonedOr = cloneMappedPureOp(nb, op, map);
    if (failed(clonedOr))
      return failure();
  }

  auto yield = dyn_cast<linalg::YieldOp>(srcBlock.getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return failure();

  Value yielded = map.lookupOrNull(yield.getOperand(0));
  if (!yielded)
    return failure();
  return yielded;
}

static FailureOr<Value>
materializeTensorSliceLikeAsSubview(OpBuilder &b, Location loc, Value v) {
  return materializeRank1InputAsSubview(b, loc, v);
}

//===----------------------------------------------------------------------===//
// Localized generic rewrite helpers
//===----------------------------------------------------------------------===//

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

static LogicalResult finalizeSuccessfulTensorToMemrefRewrite(
    mlir::hexagon::nsp::MaterializeTileOp tileOp, Operation *oldOp,
    Value initArg) {
  tileOp.erase();

  if (oldOp && oldOp->use_empty())
    oldOp->erase();

  if (auto emptyOp = initArg.getDefiningOp<tensor::EmptyOp>())
    eraseIfDead(emptyOp);

  return success();
}

//===----------------------------------------------------------------------===//
// Tensor slice matchers
//===----------------------------------------------------------------------===//

/// Return true iff `v` is the induction variable of `forOp`.
static bool isLoopIV(Value v, scf::ForOp forOp) {
  return v == forOp.getInductionVar();
}

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

/// Match a rank-reduced tensor.extract_slice taken from a rank-2 source:
///   %slice = tensor.extract_slice %source[%row, %col][1, %size][1, 1]
///            : tensor<MxNxt> to tensor<%size xt>
///
/// This is the canonical rank-reduced form of a [1 x C] row slice.
static LogicalResult
matchRankReducedRowSliceFromRank2(Value v, Value &source, OpFoldResult &rowOff,
                                  OpFoldResult &colOff,
                                  OpFoldResult &chunkSize) {
  auto extract =
      stripTensorCastWrappers(v).getDefiningOp<tensor::ExtractSliceOp>();
  if (!extract)
    return failure();

  auto sourceTy = dyn_cast<RankedTensorType>(extract.getSource().getType());
  auto resultTy = dyn_cast<RankedTensorType>(extract.getResult().getType());
  if (!sourceTy || sourceTy.getRank() != 2)
    return failure();
  if (!resultTy || resultTy.getRank() != 1)
    return failure();

  auto mixedOffsets = extract.getMixedOffsets();
  auto mixedSizes = extract.getMixedSizes();
  auto mixedStrides = extract.getMixedStrides();
  if (mixedOffsets.size() != 2 || mixedSizes.size() != 2 ||
      mixedStrides.size() != 2)
    return failure();

  auto size0Attr = dyn_cast<Attribute>(mixedSizes[0]);
  auto stride0Attr = dyn_cast<Attribute>(mixedStrides[0]);
  auto stride1Attr = dyn_cast<Attribute>(mixedStrides[1]);
  if (!size0Attr || !stride0Attr || !stride1Attr)
    return failure();
  if (cast<IntegerAttr>(size0Attr).getInt() != 1 ||
      cast<IntegerAttr>(stride0Attr).getInt() != 1 ||
      cast<IntegerAttr>(stride1Attr).getInt() != 1)
    return failure();

  source = extract.getSource();
  rowOff = mixedOffsets[0];
  colOff = mixedOffsets[1];
  chunkSize = mixedSizes[1];
  return success();
}

/// Match a rank-reduced tensor.extract_slice taken from a rank-1 source:
///   %slice = tensor.extract_slice %source[%i] [1] [1]
///            : tensor<Mxt> to tensor<t>
///
/// This is the canonical rank-reduced form of a scalar slice from a rank-1
/// tensor. It is used by broadcasted row-wise operands after localization.
static LogicalResult
matchRankReducedScalarSliceFromRank1(Value v, Value &source,
                                     OpFoldResult &offset) {
  auto extract =
      stripTensorCastWrappers(v).getDefiningOp<tensor::ExtractSliceOp>();
  if (!extract)
    return failure();

  auto sourceTy = dyn_cast<RankedTensorType>(extract.getSource().getType());
  auto resultTy = dyn_cast<RankedTensorType>(extract.getResult().getType());
  if (!sourceTy || sourceTy.getRank() != 1)
    return failure();
  if (!resultTy || resultTy.getRank() != 0)
    return failure();

  auto mixedOffsets = extract.getMixedOffsets();
  auto mixedSizes = extract.getMixedSizes();
  auto mixedStrides = extract.getMixedStrides();
  if (mixedOffsets.size() != 1 || mixedSizes.size() != 1 ||
      mixedStrides.size() != 1)
    return failure();

  auto sizeAttr = dyn_cast<Attribute>(mixedSizes[0]);
  auto strideAttr = dyn_cast<Attribute>(mixedStrides[0]);
  if (!sizeAttr || !strideAttr)
    return failure();
  if (cast<IntegerAttr>(sizeAttr).getInt() != 1 ||
      cast<IntegerAttr>(strideAttr).getInt() != 1)
    return failure();

  source = extract.getSource();
  offset = mixedOffsets[0];
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

//===----------------------------------------------------------------------===//
// Rank-1 loop fast-path
//===----------------------------------------------------------------------===//

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

    if (succeeded(cloneMappedPureOp(nb, op, map)))
      continue;

    Operation *cloned = nb.clone(op, map);
    for (auto it : llvm::zip(op.getResults(), cloned->getResults()))
      map.map(std::get<0>(it), std::get<1>(it));
  }

  nb.create<scf::YieldOp>(loc);
  return success();
}

//===----------------------------------------------------------------------===//
// Rank-2 loop fast-path
//===----------------------------------------------------------------------===//

/// Try to rewrite the current elemenwise_2d-style pattern:
///   - outer tile / accumulator is rank-2
///   - inner vectorized chunks are rank-2 slices of shape [1 x C]
///
/// Expected inner shape:
///   %a = tensor.extract_slice %in0[%i, %j] [1, C] [1, 1]
///         : tensor<T0xT1xf32> to tensor<1xCxf32>
///   %b = tensor.extract_slice %in1[%i, %j] [1, C] [1, 1]
///         : tensor<T0xT1xf32> to tensor<1xCxf32>
///   %o = tensor.extract_slice %acc[%i, %j] [1, C] [1, 1]
///         : tensor<T0xT1xf32> to tensor<1xCxf32>
///   %ra = vector.transfer_read %a[%c0, %c0], ...
///   %rb = vector.transfer_read %b[%c0, %c0], ...
///   %tw = vector.transfer_write %vec, %o[%c0, %c0], ...
///   %next = tensor.insert_slice %tw into %acc[%i, %j] [1, C] [1, 1]
///
/// Rewritten form:
///   - materialize each input tile as a rank-2 memref subview
///   - inside the nested loops, build rank-2 chunk views [1 x C]
///   - write directly into a rank-2 chunk view over the destination subview
static LogicalResult tryRewriteRank2TileRank1ChunkToMemref(
    OpBuilder &b, Location loc, scf::ForOp outerFor, Value destSubview,
    mlir::hexagon::nsp::MaterializeTileOp tileOp) {

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

  auto insertedChunkTy =
      dyn_cast<RankedTensorType>(insertSlice.getSource().getType());
  if (!insertedChunkTy)
    return failure();

  const bool chunkIsRank2 = insertedChunkTy.getRank() == 2;
  const bool chunkIsRank1 = insertedChunkTy.getRank() == 1;
  if (!chunkIsRank1 && !chunkIsRank2) {
    tileOp.emitRemark()
        << "NSPMaterialize: rank2 fast-path rejected; expected the inner "
           "vectorized chunk to be either tensor<1xC> or rank-reduced "
           "tensor<C>";
    return failure();
  }

  if (chunkIsRank2) {
    if (!ShapedType::isDynamic(insertedChunkTy.getShape()[0]) &&
        insertedChunkTy.getShape()[0] != 1)
      return failure();
  }

  auto insertOffsets = insertSlice.getMixedOffsets();
  auto insertSizes = insertSlice.getMixedSizes();
  auto insertStrides = insertSlice.getMixedStrides();
  if (insertOffsets.size() != 2 || insertSizes.size() != 2 ||
      insertStrides.size() != 2)
    return failure();

  auto insRowOff = dyn_cast<Value>(insertOffsets[0]);
  auto insColOff = dyn_cast<Value>(insertOffsets[1]);
  if (!insRowOff || !insColOff)
    return failure();
  if (!isLoopIV(insRowOff, outerFor) || !isLoopIV(insColOff, innerFor))
    return failure();

  auto size0Attr = dyn_cast<Attribute>(insertSizes[0]);
  auto stride0Attr = dyn_cast<Attribute>(insertStrides[0]);
  auto stride1Attr = dyn_cast<Attribute>(insertStrides[1]);
  if (!size0Attr || !stride0Attr || !stride1Attr)
    return failure();

  auto size0Int = dyn_cast<IntegerAttr>(size0Attr);
  auto stride0Int = dyn_cast<IntegerAttr>(stride0Attr);
  auto stride1Int = dyn_cast<IntegerAttr>(stride1Attr);
  if (!size0Int || !stride0Int || !stride1Int)
    return failure();

  // Expect insert_slice [1, C] [1, 1].
  if (size0Int.getInt() != 1 || stride0Int.getInt() != 1 ||
      stride1Int.getInt() != 1)
    return failure();

  auto transferWrite =
      insertSlice.getSource().getDefiningOp<vector::TransferWriteOp>();
  if (!transferWrite)
    return failure();

  Value accChunkSource;
  OpFoldResult accColOff;
  OpFoldResult accChunkSize;

  if (chunkIsRank2) {
    SmallVector<OpFoldResult> accChunkOffsets, accChunkSizes, accChunkStrides;
    if (failed(matchRank2TensorExtractSlice(transferWrite.getBase(),
                                            accChunkSource, accChunkOffsets,
                                            accChunkSizes, accChunkStrides)))
      return failure();

    if (accChunkSource != innerBody->getArgument(1))
      return failure();
    if (accChunkOffsets.size() != 2 || accChunkSizes.size() != 2 ||
        accChunkStrides.size() != 2)
      return failure();

    auto accRowOff = dyn_cast<Value>(accChunkOffsets[0]);
    auto accColOffVal = dyn_cast<Value>(accChunkOffsets[1]);
    if (!accRowOff || !accColOffVal)
      return failure();
    if (!isLoopIV(accRowOff, outerFor) || !isLoopIV(accColOffVal, innerFor))
      return failure();

    auto accSize0Attr = dyn_cast<Attribute>(accChunkSizes[0]);
    auto accStride0Attr = dyn_cast<Attribute>(accChunkStrides[0]);
    auto accStride1Attr = dyn_cast<Attribute>(accChunkStrides[1]);
    if (!accSize0Attr || !accStride0Attr || !accStride1Attr)
      return failure();

    auto accSize0Int = dyn_cast<IntegerAttr>(accSize0Attr);
    auto accStride0Int = dyn_cast<IntegerAttr>(accStride0Attr);
    auto accStride1Int = dyn_cast<IntegerAttr>(accStride1Attr);
    if (!accSize0Int || !accStride0Int || !accStride1Int)
      return failure();
    if (accSize0Int.getInt() != 1 || accStride0Int.getInt() != 1 ||
        accStride1Int.getInt() != 1)
      return failure();

    accColOff = accChunkOffsets[1];
    accChunkSize = accChunkSizes[1];
  } else {
    OpFoldResult accRowOff, accColOffTmp, accChunkSizeTmp;
    if (failed(matchRankReducedRowSliceFromRank2(
            transferWrite.getBase(), accChunkSource, accRowOff, accColOffTmp,
            accChunkSizeTmp)))
      return failure();

    if (accChunkSource != innerBody->getArgument(1))
      return failure();

    auto accRowOffVal = dyn_cast<Value>(accRowOff);
    auto accColOffVal = dyn_cast<Value>(accColOffTmp);
    if (!accRowOffVal || !accColOffVal)
      return failure();
    if (!isLoopIV(accRowOffVal, outerFor) || !isLoopIV(accColOffVal, innerFor))
      return failure();

    accColOff = accColOffTmp;
    accChunkSize = accChunkSizeTmp;
  }

  enum class InputReadKind {
    Rank2Chunk,
    Rank1Scalar,
  };
  struct InputReadSpec {
    InputReadKind kind;
    Value sourceTensor;
    OpFoldResult colOff;
    OpFoldResult chunkSize;
    OpFoldResult scalarOff;
  };

  SmallVector<vector::TransferReadOp> transferReads;
  for (Operation &op : innerBody->without_terminator()) {
    if (auto tr = dyn_cast<vector::TransferReadOp>(op))
      transferReads.push_back(tr);
  }
  if (transferReads.empty() || transferReads.size() > 2)
    return failure();

  SmallVector<InputReadSpec> readSpecs;
  readSpecs.reserve(transferReads.size());

  for (vector::TransferReadOp tr : transferReads) {
    Value srcTileTensor;

    if (chunkIsRank2) {
      SmallVector<OpFoldResult> chunkOffsets, chunkSizes, chunkStrides;
      if (succeeded(matchRank2TensorExtractSlice(tr.getBase(), srcTileTensor,
                                                 chunkOffsets, chunkSizes,
                                                 chunkStrides))) {
        if (chunkOffsets.size() != 2 || chunkSizes.size() != 2 ||
            chunkStrides.size() != 2)
          return failure();

        auto rowOff = dyn_cast<Value>(chunkOffsets[0]);
        auto colOff = dyn_cast<Value>(chunkOffsets[1]);
        if (!rowOff || !colOff)
          return failure();
        if (!isLoopIV(rowOff, outerFor) || !isLoopIV(colOff, innerFor))
          return failure();

        auto size0Attr = dyn_cast<Attribute>(chunkSizes[0]);
        auto stride0Attr = dyn_cast<Attribute>(chunkStrides[0]);
        auto stride1Attr = dyn_cast<Attribute>(chunkStrides[1]);
        if (!size0Attr || !stride0Attr || !stride1Attr)
          return failure();

        auto size0Int = dyn_cast<IntegerAttr>(size0Attr);
        auto stride0Int = dyn_cast<IntegerAttr>(stride0Attr);
        auto stride1Int = dyn_cast<IntegerAttr>(stride1Attr);
        if (!size0Int || !stride0Int || !stride1Int)
          return failure();

        if (size0Int.getInt() != 1 || stride0Int.getInt() != 1 ||
            stride1Int.getInt() != 1)
          return failure();

        readSpecs.push_back(InputReadSpec{InputReadKind::Rank2Chunk,
                                          srcTileTensor, chunkOffsets[1],
                                          chunkSizes[1], OpFoldResult()});
        continue;
      }
    } else {
      OpFoldResult rowOff, colOff, chunkSize;
      if (succeeded(matchRankReducedRowSliceFromRank2(
              tr.getBase(), srcTileTensor, rowOff, colOff, chunkSize))) {
        auto rowOffVal = dyn_cast<Value>(rowOff);
        auto colOffVal = dyn_cast<Value>(colOff);
        if (!rowOffVal || !colOffVal)
          return failure();
        if (!isLoopIV(rowOffVal, outerFor) || !isLoopIV(colOffVal, innerFor))
          return failure();

        readSpecs.push_back(InputReadSpec{InputReadKind::Rank2Chunk,
                                          srcTileTensor, colOff, chunkSize,
                                          OpFoldResult()});
        continue;
      }
    }

    // New case: rank-1 scalar broadcast.
    OpFoldResult scalarOff;
    if (failed(matchRankReducedScalarSliceFromRank1(tr.getBase(), srcTileTensor,
                                                    scalarOff)))
      return failure();

    auto scalarOffVal = dyn_cast<Value>(scalarOff);
    if (!scalarOffVal)
      return failure();
    if (!isLoopIV(scalarOffVal, outerFor))
      return failure();

    // Current vectorized scalar-broadcast form reads a scalar tensor with no
    // explicit indices and broadcasts it via permutation_map = () -> (0).
    if (!tr.getIndices().empty())
      return failure();
    if (tr.getMask())
      return failure();

    readSpecs.push_back(InputReadSpec{InputReadKind::Rank1Scalar, srcTileTensor,
                                      OpFoldResult(), OpFoldResult(),
                                      scalarOff});
  }

  SmallVector<Value> inputTileMemrefs;
  inputTileMemrefs.reserve(readSpecs.size());
  for (const InputReadSpec &spec : readSpecs) {
    FailureOr<Value> memrefOr = failure();
    if (spec.kind == InputReadKind::Rank2Chunk)
      memrefOr = materializeRank2InputAsSubview(b, loc, spec.sourceTensor);
    else
      memrefOr = materializeRank1InputAsSubview(b, loc, spec.sourceTensor);
    if (failed(memrefOr))
      return failure();
    inputTileMemrefs.push_back(*memrefOr);
    // }
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

  // Build rank-2 chunk memrefs [1 x C] from the rank-2 input tiles.
  SmallVector<Value> newChunkMemrefs;
  newChunkMemrefs.reserve(readSpecs.size());
  for (size_t i = 0; i < readSpecs.size(); ++i) {
    const InputReadSpec &spec = readSpecs[i];

    if (spec.kind == InputReadKind::Rank1Scalar) {
      newChunkMemrefs.push_back(Value());
      continue;
    }

    FailureOr<Value> chunkOr = failure();
    if (chunkIsRank2) {
      SmallVector<OpFoldResult> offs = {
          newOuter.getInductionVar(),
          map.lookupOrDefault(dyn_cast_if_present<Value>(spec.colOff))};
      SmallVector<OpFoldResult> sizes = {ib.getIndexAttr(1), spec.chunkSize};
      SmallVector<OpFoldResult> strides = {ib.getIndexAttr(1),
                                           ib.getIndexAttr(1)};
      chunkOr =
          buildRank2Subview(ib, loc, inputTileMemrefs[i], offs, sizes, strides);
    } else {
      chunkOr = buildRank1SubviewFromRank2(
          ib, loc, inputTileMemrefs[i], newOuter.getInductionVar(),
          map.lookupOrDefault(dyn_cast_if_present<Value>(spec.colOff)),
          spec.chunkSize);
    }

    if (failed(chunkOr))
      return failure();
    newChunkMemrefs.push_back(*chunkOr);
  }

  FailureOr<Value> outChunkOr = failure();
  if (chunkIsRank2) {
    SmallVector<OpFoldResult> outOffsets = {newOuter.getInductionVar(),
                                            newInner.getInductionVar()};
    SmallVector<OpFoldResult> outSizes = {ib.getIndexAttr(1), accChunkSize};
    SmallVector<OpFoldResult> outStrides = {ib.getIndexAttr(1),
                                            ib.getIndexAttr(1)};
    outChunkOr = buildRank2Subview(ib, loc, destSubview, outOffsets, outSizes,
                                   outStrides);
  } else {
    outChunkOr = buildRank1SubviewFromRank2(
        ib, loc, destSubview, newOuter.getInductionVar(),
        newInner.getInductionVar(), accChunkSize);
  }
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

      const InputReadSpec &spec = readSpecs[idx];
      if (spec.kind == InputReadKind::Rank1Scalar) {
        Value scalarIndex =
            map.lookupOrDefault(dyn_cast_if_present<Value>(spec.scalarOff));
        if (!scalarIndex)
          return failure();

        auto scalarMemrefTy =
            dyn_cast<MemRefType>(inputTileMemrefs[idx].getType());
        if (!scalarMemrefTy || scalarMemrefTy.getRank() != 1)
          return failure();

        Value scalar = ib.create<memref::LoadOp>(
            tr.getLoc(), inputTileMemrefs[idx], ValueRange{scalarIndex});
        auto splat =
            ib.create<vector::SplatOp>(tr.getLoc(), tr.getVectorType(), scalar);
        map.map(tr.getResult(), splat.getResult());
      } else {
        SmallVector<Value> indices;
        indices.reserve(tr.getIndices().size());
        for (Value iv : tr.getIndices())
          indices.push_back(map.lookupOrDefault(iv));

        auto newRead = ib.create<vector::TransferReadOp>(
            tr.getLoc(), tr.getVectorType(), newChunkMemrefs[idx], indices,
            tr.getPermutationMapAttr(), tr.getPadding(), tr.getMask(),
            tr.getInBoundsAttr());
        map.map(tr.getResult(), newRead.getResult());
      }

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

    if (succeeded(cloneMappedPureOp(ib, op, map)))
      continue;

    Operation *cloned = ib.clone(op, map);
    for (auto it : llvm::zip(op.getResults(), cloned->getResults()))
      map.map(std::get<0>(it), std::get<1>(it));
  }

  ib.create<scf::YieldOp>(loc);
  ob.create<scf::YieldOp>(loc);
  return success();
}

//===----------------------------------------------------------------------===//
// Tile materialization driver
//===----------------------------------------------------------------------===//

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
        auto subviewOr = materializeRank1InputAsSubview(b, loc, in->get());
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
          return finalizeSuccessfulTensorToMemrefRewrite(tileOp, oldGenericOp,
                                                         oldInit);
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
      if (succeeded(tryRewriteRank2TileRank1ChunkToMemref(
              b, loc, tileLoop, destSubview, tileOp))) {
        return finalizeSuccessfulTensorToMemrefRewrite(tileOp, oldLoopOp,
                                                       initArg);
      }
      // tileOp.emitRemark()
      //     << "NSPMaterialize: rank2/rank1 fast-path did not match";
    }

    // Keep the existing rank-1 fast-path unchanged.
    // If rank-2 matching fails, we still fall back to the old logic below.
    if (succeeded(
            tryRewriteScfForTileToMemref(b, loc, tileLoop, destSubview))) {
      return finalizeSuccessfulTensorToMemrefRewrite(tileOp, oldLoopOp,
                                                     initArg);
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

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

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
