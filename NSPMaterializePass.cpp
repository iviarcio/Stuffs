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
//   - the generic tile hand-off supports rank-1 through rank-4 destination
//     materialization through tile_shape,
//   - split_axis is honored when building destination/input tile subviews,
//   - the direct memref rewrite for localized identity linalg.generic supports
//     ranks up to 4, including rank-reduced canonicalized rank-3 views,
//   - a dedicated fast-path exists for the elementwise 2d-style pattern:
//       * rank-2 outer tile / accumulator
//       * rank-1 inner vectorized chunks
//   - unsupported tiled-loop patterns still fall back to
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

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
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
// Forward Declarations
//===----------------------------------------------------------------------===//

static LogicalResult matchRankReducedScalarSliceFromRank1(Value v,
                                                          Value &source,
                                                          OpFoldResult &offset);

static FailureOr<Value> materializeScalarLikeValueAtIndexImpl(
    OpBuilder &b, Location loc, Value v, Value index,
    llvm::SmallPtrSetImpl<Operation *> &stack, unsigned depth);

//===----------------------------------------------------------------------===//
// Grid / ABI helpers
//===----------------------------------------------------------------------===//

static Value castToIndexIfNeeded(Value v, OpBuilder &b, Location loc) {
  if (v.getType().isIndex())
    return v;
  if (isa<IntegerType>(v.getType()))
    return arith::IndexCastOp::create(b, loc, b.getIndexType(), v);
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

  Value mul = arith::MulIOp::create(b, loc, cid, ntpc);
  return arith::AddIOp::create(b, loc, mul, tid).getResult();
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

/// Return the semantic source value after stripping tensor.cast and
/// bufferization.alloc_tensor copy wrappers.
///
/// Canonicalization/bufferization preparation may turn a localized tensor value
/// into:
///   %copy = bufferization.alloc_tensor() copy(%value)
///
/// That copy is a tensor materialization artifact.  The NSP memref fast-path
/// should look through it when it is trying to recover the original producer or
/// the original slice-like input.  Plain alloc_tensor without a copy is not
/// stripped because it represents an init/temporary value, not an alias to a
/// source tensor.
static Value stripTensorMaterializationWrappers(Value v) {
  Value cur = stripTensorCastWrappers(v);

  while (auto allocTensor = cur.getDefiningOp<bufferization::AllocTensorOp>()) {
    Value copy = allocTensor.getCopy();
    if (!copy)
      break;
    cur = stripTensorCastWrappers(copy);
  }

  return cur;
}

static constexpr int64_t kMaxSupportedIdentityElementwiseRank = 4;

static bool isSupportedIdentityElementwiseRank(int64_t rank) {
  return rank >= 1 && rank <= kMaxSupportedIdentityElementwiseRank;
}

/// Return true iff `g` is a localized identity-map elementwise generic that can
/// be safely rewritten from tensor semantics to memref semantics.
///
/// Current pattern constraints:
///   - carries the `nsp.localized` marker
///   - 1 to 4 DPS inputs
///   - exactly 1 DPS init/output
///   - ranked tensor shape with rank in [1, 4]
///   - all-parallel iterator space
///   - identity indexing maps
static bool isSimpleLocalizedElementwiseGeneric(linalg::GenericOp g) {
  if (!g || !g->hasAttr("nsp.localized"))
    return false;

  const int64_t numInputs = g.getNumDpsInputs();
  if (numInputs < 1 || numInputs > 4 || g.getNumDpsInits() != 1)
    return false;

  auto resultTy = dyn_cast<RankedTensorType>(g.getResult(0).getType());
  if (!resultTy || !isSupportedIdentityElementwiseRank(resultTy.getRank()))
    return false;

  const int64_t rank = resultTy.getRank();
  if (g.getNumLoops() != rank)
    return false;

  auto iters = g.getIteratorTypesArray();
  if (static_cast<int64_t>(iters.size()) != rank)
    return false;
  if (!llvm::all_of(iters, [](mlir::utils::IteratorType it) {
        return it == mlir::utils::IteratorType::parallel;
      }))
    return false;

  auto maps = g.getIndexingMapsArray();
  if (static_cast<int64_t>(maps.size()) != numInputs + 1)
    return false;

  for (AffineMap m : maps)
    if (!m.isIdentity())
      return false;

  for (OpOperand *in : g.getDpsInputOperands()) {
    auto inTy = dyn_cast<RankedTensorType>(in->get().getType());
    if (!inTy || inTy.getRank() != rank)
      return false;
    if (inTy.getShape() != resultTy.getShape())
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

/// Return true iff `t` is either:
///   - a scalar SSA type, or
///   - a ranked tensor of rank 0 (tensor<elemTy>)
static bool isScalarLikeType(Type t) {
  if (!t)
    return false;
  if (!isa<ShapedType>(t))
    return true;
  auto st = dyn_cast<ShapedType>(t);
  return st && st.hasRank() && st.getRank() == 0;
}

/// Scalarize a type for elementwise cloning:
///   - tensor<...xT> -> T   (only for ranked rank-0 / rank-1 use here)
///   - scalar types stay unchanged
static Type getScalarizedType(Type t) {
  if (auto st = dyn_cast<ShapedType>(t)) {
    if (st.hasRank() && (st.getRank() == 0 || st.getRank() == 1))
      return st.getElementType();
  }
  return t;
}

/// Clone a memory-effect-free op using already-materialized scalar operands,
/// but scalarize ranked rank-0 / rank-1 tensor result types to their element
/// type. This is used when rebuilding short arith/math chains that originally
/// operated on tensor<...> values, but in the memref fast-path should operate
/// on scalars.
static FailureOr<Operation *>
clonePureOpWithScalarizedTypes(OpBuilder &b, Operation &op,
                               ValueRange operands) {
  if (op.getNumRegions() != 0)
    return failure();
  if (!isMemoryEffectFree(&op))
    return failure();

  OperationState state(op.getLoc(), op.getName());
  state.addOperands(operands);

  SmallVector<Type> resultTypes;
  resultTypes.reserve(op.getNumResults());
  for (Type t : op.getResultTypes())
    resultTypes.push_back(getScalarizedType(t));
  state.addTypes(resultTypes);
  state.addAttributes(op.getAttrs());

  return b.create(state);
}

//===----------------------------------------------------------------------===//
// Memref view builders
//===----------------------------------------------------------------------===//

/// Build a ranked memref.subview at the current insertion point.
static FailureOr<Value> buildRankedSubview(OpBuilder &b, Location loc,
                                           Value baseMemref,
                                           ArrayRef<OpFoldResult> offsets,
                                           ArrayRef<OpFoldResult> sizes,
                                           ArrayRef<OpFoldResult> strides) {
  auto baseTy = dyn_cast<MemRefType>(baseMemref.getType());
  if (!baseTy || baseTy.getRank() < 1)
    return failure();

  const int64_t rank = baseTy.getRank();
  if ((int64_t)offsets.size() != rank || (int64_t)sizes.size() != rank ||
      (int64_t)strides.size() != rank)
    return failure();

  auto subview =
      memref::SubViewOp::create(b, loc, baseMemref, offsets, sizes, strides);
  return subview.getResult();
}

/// Build a rank-reduced memref.collapse_shape view corresponding to a
/// tensor.collapse_shape / tensor.expand_shape reassociation.
///
/// The collapsed view is used by the fast-path when canonicalization removes a
/// leading unit dimension from a localized rank-4 tile, for example:
///   tensor<1x16x512x4xf16> -> tensor<16x512x4xf16>
///
/// Keep the layout fully dynamic.  The source subview already carries the
/// precise offset/stride information, and the downstream linalg.generic only
/// requires a ranked strided memref view with the right element type and rank.
static FailureOr<Value>
buildCollapsedMemrefView(OpBuilder &b, Location loc, Value sourceMemref,
                         ArrayRef<ReassociationIndices> reassociation,
                         RankedTensorType collapsedTensorTy) {
  auto sourceMemrefTy = dyn_cast<MemRefType>(sourceMemref.getType());
  if (!sourceMemrefTy || !collapsedTensorTy)
    return failure();

  if (sourceMemrefTy.getElementType() != collapsedTensorTy.getElementType())
    return failure();

  const int64_t collapsedRank = collapsedTensorTy.getRank();
  if (collapsedRank < 1 || collapsedRank >= sourceMemrefTy.getRank())
    return failure();

  if ((int64_t)reassociation.size() != collapsedRank)
    return failure();

  SmallVector<int64_t> collapsedShape(collapsedTensorTy.getShape().begin(),
                                      collapsedTensorTy.getShape().end());
  SmallVector<int64_t> collapsedStrides(collapsedRank, ShapedType::kDynamic);
  auto layout = StridedLayoutAttr::get(sourceMemrefTy.getContext(),
                                       /*offset=*/ShapedType::kDynamic,
                                       /*strides=*/collapsedStrides);

  auto collapsedMemrefTy =
      MemRefType::get(collapsedShape, sourceMemrefTy.getElementType(), layout,
                      sourceMemrefTy.getMemorySpace());

  auto collapse = memref::CollapseShapeOp::create(b, loc, collapsedMemrefTy,
                                                  sourceMemref, reassociation);
  return collapse.getResult();
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
  return buildRankedSubview(b, loc, baseMemref, offsets, sizes, strides);
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
      memref::SubViewOp::create(b, loc, baseMemref, offsets, sizes, strides);

  auto collapsedOp = memref::CollapseShapeOp::create(
      b, loc, subviewTy, full2D.getResult(), ReassociationIndices{{0, 1}});
  return collapsedOp.getResult();
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

static FailureOr<int64_t> getAllSliceAxis(Operation *op, int64_t rank) {
  auto axisAttr = op->getAttrOfType<IntegerAttr>("slice_axis");
  if (!axisAttr)
    return failure();

  int64_t axis = axisAttr.getInt();
  if (axis < 0 || axis >= rank)
    return failure();
  return axis;
}

/// Materialize a ranked memref view from a tensor slice-like value.
///
/// Supported sources:
///   - shard.all_slice over bufferization.to_tensor(memref)
///   - same-rank tensor.extract_slice over bufferization.to_tensor(memref)
///
/// The resulting memref rank matches the tensor rank. This is the generic path
/// used by localized identity elementwise kernels up to rank 4. Specialized
/// rank-1/rank-2 helpers below remain in place for vectorized tiled-loop
/// fast-paths that require additional rank-reduced matching.
static FailureOr<Value>
materializeRankedInputAsSubview(OpBuilder &b, Location loc, Value inputTensor) {
  Value base = stripTensorMaterializationWrappers(inputTensor);

  // Canonicalization may fold a leading unit dimension away from the localized
  // rank-4 tile before materialization:
  //   tensor.collapse_shape %tile [[0, 1], [2], [3]]
  //     : tensor<1x16x512x4xf16> into tensor<16x512x4xf16>
  // Materialize the source tile as a memref subview first, then build the
  // equivalent memref.collapse_shape view so the rank-3 linalg.generic can be
  // rewritten directly to memref semantics.
  if (auto collapse = base.getDefiningOp<tensor::CollapseShapeOp>()) {
    auto resultTy = dyn_cast<RankedTensorType>(collapse.getResult().getType());
    if (!resultTy || !isSupportedIdentityElementwiseRank(resultTy.getRank()))
      return failure();

    auto sourceSubviewOr =
        materializeRankedInputAsSubview(b, loc, collapse->getOperand(0));
    if (failed(sourceSubviewOr))
      return failure();

    SmallVector<ReassociationIndices> reassociation =
        collapse.getReassociationIndices();
    return buildCollapsedMemrefView(b, loc, *sourceSubviewOr, reassociation,
                                    resultTy);
  }

  Value sourceTensor;
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  int64_t rank = -1;

  if (auto allSlice = base.getDefiningOp<mlir::shard::AllSliceOp>()) {
    sourceTensor = allSlice->getOperand(0);

    auto resultTy = dyn_cast<RankedTensorType>(allSlice.getResult().getType());
    if (!resultTy || !isSupportedIdentityElementwiseRank(resultTy.getRank()))
      return failure();

    rank = resultTy.getRank();

    auto sliceAxisOr = getAllSliceAxis(allSlice.getOperation(), rank);
    if (failed(sliceAxisOr))
      return failure();
    int64_t sliceAxis = *sliceAxisOr;

    auto func = allSlice->getParentOfType<func::FuncOp>();
    if (!func)
      return failure();

    auto linearIdxOrFail = computeLinearIdxFromFuncArgs(func, b, loc);
    if (failed(linearIdxOrFail))
      return failure();

    int64_t tileExtent = resultTy.getDimSize(sliceAxis);
    if (ShapedType::isDynamic(tileExtent) || tileExtent <= 0)
      return failure();

    Value tileExtentVal = arith::ConstantIndexOp::create(b, loc, tileExtent);
    Value offset =
        arith::MulIOp::create(b, loc, *linearIdxOrFail, tileExtentVal);

    offsets.reserve(rank);
    sizes.reserve(rank);
    strides.reserve(rank);
    for (int64_t dim = 0; dim < rank; ++dim) {
      int64_t extent = resultTy.getDimSize(dim);
      if (ShapedType::isDynamic(extent) || extent <= 0)
        return failure();

      offsets.push_back(dim == sliceAxis ? OpFoldResult(offset)
                                         : OpFoldResult(b.getIndexAttr(0)));
      sizes.push_back(b.getIndexAttr(extent));
      strides.push_back(b.getIndexAttr(1));
    }
  } else if (auto extract = base.getDefiningOp<tensor::ExtractSliceOp>()) {
    sourceTensor = extract.getSource();

    auto sourceTy = dyn_cast<RankedTensorType>(sourceTensor.getType());
    auto resultTy = dyn_cast<RankedTensorType>(extract.getResult().getType());
    if (!sourceTy || !resultTy ||
        !isSupportedIdentityElementwiseRank(resultTy.getRank()))
      return failure();

    // This generic helper only handles same-rank slices. Rank-reduced slice
    // forms are handled by the existing rank-specialized helpers.
    if (sourceTy.getRank() != resultTy.getRank())
      return failure();

    rank = resultTy.getRank();

    auto mixedOffsets = extract.getMixedOffsets();
    auto mixedSizes = extract.getMixedSizes();
    auto mixedStrides = extract.getMixedStrides();
    if ((int64_t)mixedOffsets.size() != rank ||
        (int64_t)mixedSizes.size() != rank ||
        (int64_t)mixedStrides.size() != rank)
      return failure();

    offsets.assign(mixedOffsets.begin(), mixedOffsets.end());
    sizes.assign(mixedSizes.begin(), mixedSizes.end());
    strides.assign(mixedStrides.begin(), mixedStrides.end());
  } else {
    return failure();
  }

  Value baseTensor = stripTensorMaterializationWrappers(sourceTensor);
  auto toTensor = baseTensor.getDefiningOp<bufferization::ToTensorOp>();
  if (!toTensor)
    return failure();

  Value sourceMemref = toTensor->getOperand(0);
  auto sourceMemrefTy = dyn_cast<MemRefType>(sourceMemref.getType());
  if (!sourceMemrefTy || sourceMemrefTy.getRank() != rank)
    return failure();

  return buildRankedSubview(b, loc, sourceMemref, offsets, sizes, strides);
}

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

    auto sliceAxisOr = getAllSliceAxis(allSlice.getOperation(), /*rank=*/2);
    if (failed(sliceAxisOr))
      return failure();
    int64_t sliceAxis = *sliceAxisOr;

    auto func = allSlice->getParentOfType<func::FuncOp>();
    if (!func)
      return failure();

    auto linearIdxOrFail = computeLinearIdxFromFuncArgs(func, b, loc);
    if (failed(linearIdxOrFail))
      return failure();

    int64_t tileExtent = resultTy.getShape()[sliceAxis];
    Value tileExtentVal = arith::ConstantIndexOp::create(b, loc, tileExtent);
    Value offset =
        arith::MulIOp::create(b, loc, *linearIdxOrFail, tileExtentVal);

    offsets = {sliceAxis == 0 ? OpFoldResult(offset)
                              : OpFoldResult(b.getIndexAttr(0)),
               sliceAxis == 1 ? OpFoldResult(offset)
                              : OpFoldResult(b.getIndexAttr(0))};
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

      auto sliceAxisOr = getAllSliceAxis(allSliceOp.getOperation(), /*rank=*/2);
      if (failed(sliceAxisOr))
        return failure();
      int64_t sliceAxis = *sliceAxisOr;

      auto func = allSliceOp->getParentOfType<func::FuncOp>();
      if (!func)
        return failure();

      auto linearIdxOrFail = computeLinearIdxFromFuncArgs(func, b, loc);
      if (failed(linearIdxOrFail))
        return failure();

      int64_t tileExtent = resultTy.getShape()[sliceAxis];
      Value tileExtentVal = arith::ConstantIndexOp::create(b, loc, tileExtent);
      Value offset =
          arith::MulIOp::create(b, loc, *linearIdxOrFail, tileExtentVal);

      // Compose the existing offsets with the all_slice tile offset.
      if (offsets.empty()) {
        offsets = {sliceAxis == 0 ? OpFoldResult(offset)
                                  : OpFoldResult(b.getIndexAttr(0)),
                   sliceAxis == 1 ? OpFoldResult(offset)
                                  : OpFoldResult(b.getIndexAttr(0))};
        sizes = {b.getIndexAttr(resultTy.getShape()[0]),
                 b.getIndexAttr(resultTy.getShape()[1])};
        strides = {b.getIndexAttr(1), b.getIndexAttr(1)};
      } else {
        OpFoldResult &old = offsets[sliceAxis];
        if (auto oldVal = dyn_cast<Value>(old)) {
          Value newOffset =
              arith::AddIOp::create(b, loc, offset, oldVal).getResult();
          old = newOffset;
        } else if (auto oldAttr = dyn_cast<Attribute>(old)) {
          Value oldVal = arith::ConstantIndexOp::create(
                             b, loc, cast<IntegerAttr>(oldAttr).getInt())
                             .getResult();
          Value newOffset =
              arith::AddIOp::create(b, loc, offset, oldVal).getResult();
          old = newOffset;
        } else {
          return failure();
        }
      }

      baseTensor = allSliceOp.getOperand();
      continue;
    }
    break;
  }

  auto toTensor = baseTensor.getDefiningOp<bufferization::ToTensorOp>();
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

    Value sliceSizeVal = arith::ConstantIndexOp::create(b, loc, sliceSize);
    Value off = arith::MulIOp::create(b, loc, *linearIdxOrFail, sliceSizeVal);

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

  return memref::SubViewOp::create(b, loc, subviewTy, sourceMemref, offsets,
                                   sizes, strides)
      .getResult();
}

/// Convert an OpFoldResult known to represent an index into a Value.
static FailureOr<Value> materializeIndexFromOFR(OpBuilder &b, Location loc,
                                                OpFoldResult ofr) {
  if (auto v = dyn_cast<Value>(ofr))
    return castToIndexIfNeeded(v, b, loc);

  auto attr = dyn_cast<Attribute>(ofr);
  if (!attr)
    return failure();

  auto intAttr = dyn_cast<IntegerAttr>(attr);
  if (!intAttr)
    return failure();

  return arith::ConstantIndexOp::create(b, loc, intAttr.getInt()).getResult();
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
///
/// This helper supports:
///   - direct rank-1 local tensor sources backed by memref
///   - simple localized rank-1 linalg.generic chains
///   - short pure rank-1 / rank-0 arith/math chains
///
/// The recursion is bounded and cycle-checked.
static FailureOr<Value>
materializeScalarFromRank1TensorAtIndex(OpBuilder &b, Location loc,
                                        Value tensorLike, Value index) {
  llvm::SmallPtrSet<Operation *, 16> stack;
  return materializeScalarLikeValueAtIndexImpl(b, loc, tensorLike, index, stack,
                                               /*depth=*/0);
}

static FailureOr<Value> materializeScalarLikeValueAtIndexImpl(
    OpBuilder &b, Location loc, Value v, Value index,
    llvm::SmallPtrSetImpl<Operation *> &stack, unsigned depth) {
  if (depth > 16)
    return failure();

  Value base = stripTensorCastWrappers(v);
  Type baseTy = base.getType();

  // Pure scalar SSA values are already materialized.
  if (!isa<ShapedType>(baseTy))
    return base;

  auto shapedTy = dyn_cast<ShapedType>(baseTy);
  if (!shapedTy || !shapedTy.hasRank())
    return failure();

  // Case A: rank-1 tensor-like value -> scalar at `index`.
  if (shapedTy.getRank() == 1) {
    auto memrefOr = materializeRank1InputAsSubview(b, loc, base);
    if (succeeded(memrefOr)) {
      Value memref = *memrefOr;
      auto memrefTy = dyn_cast<MemRefType>(memref.getType());
      if (!memrefTy || memrefTy.getRank() != 1)
        return failure();
      return memref::LoadOp::create(b, loc, memref, ValueRange{index})
          .getResult();
    }

    Operation *def = base.getDefiningOp();
    if (!def)
      return failure();
    if (!stack.insert(def).second)
      return failure();

    auto eraseFromStack = llvm::scope_exit([&]() { stack.erase(def); });

    // Case A1: simple localized rank-1 generic.
    if (auto generic = dyn_cast<linalg::GenericOp>(def)) {
      if (!isSimpleLocalizedElementwiseGeneric(generic))
        return failure();

      auto resultTy =
          dyn_cast<RankedTensorType>(generic.getResult(0).getType());
      if (!resultTy || resultTy.getRank() != 1)
        return failure();

      const int64_t numInputs = generic.getNumDpsInputs();
      SmallVector<Value> scalarInputs;
      scalarInputs.reserve(numInputs);

      for (OpOperand *in : generic.getDpsInputOperands()) {
        auto scalarOr = materializeScalarLikeValueAtIndexImpl(
            b, loc, in->get(), index, stack, depth + 1);
        if (failed(scalarOr))
          return failure();
        scalarInputs.push_back(*scalarOr);
      }

      Block &srcBlock = generic.getRegion().front();
      if (static_cast<int64_t>(srcBlock.getNumArguments()) != numInputs + 1)
        return failure();

      // Only support bodies that do not depend on the output argument.
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

    // Case A2: short pure rank-1 chain (e.g. arith/math on rank-1 tensors).
    if (def->getNumRegions() != 0 || !isMemoryEffectFree(def) ||
        def->getNumResults() != 1)
      return failure();

    auto defResTy = dyn_cast<ShapedType>(def->getResult(0).getType());
    if (!defResTy || !defResTy.hasRank() || defResTy.getRank() != 1)
      return failure();

    SmallVector<Value> scalarOperands;
    scalarOperands.reserve(def->getNumOperands());
    for (Value operand : def->getOperands()) {
      auto scalarOr = materializeScalarLikeValueAtIndexImpl(
          b, loc, operand, index, stack, depth + 1);
      if (failed(scalarOr))
        return failure();
      scalarOperands.push_back(*scalarOr);
    }

    auto clonedOr =
        clonePureOpWithScalarizedTypes(b, *def, ValueRange{scalarOperands});
    if (failed(clonedOr))
      return failure();
    if ((*clonedOr)->getNumResults() != 1)
      return failure();
    return (*clonedOr)->getResult(0);
  }

  // Case B: rank-0 tensor-like value -> scalar.
  if (shapedTy.getRank() == 0) {
    // Common case: tensor.extract_slice rank-reduced from a rank-1 source.
    Value sourceTensor;
    OpFoldResult offset;
    if (succeeded(
            matchRankReducedScalarSliceFromRank1(base, sourceTensor, offset))) {
      auto memrefOr = materializeRank1InputAsSubview(b, loc, sourceTensor);
      if (failed(memrefOr))
        return failure();

      auto idxOr = materializeIndexFromOFR(b, loc, offset);
      if (failed(idxOr))
        return failure();

      Value memref = *memrefOr;
      auto memrefTy = dyn_cast<MemRefType>(memref.getType());
      if (!memrefTy || memrefTy.getRank() != 1)
        return failure();

      return memref::LoadOp::create(b, loc, memref, ValueRange{*idxOr})
          .getResult();
    }

    Operation *def = base.getDefiningOp();
    if (!def)
      return failure();
    if (!stack.insert(def).second)
      return failure();

    auto eraseFromStack = llvm::scope_exit([&]() { stack.erase(def); });

    // Support short pure scalar/rank-0 chains (arith/math/etc.).
    if (def->getNumRegions() != 0 || !isMemoryEffectFree(def) ||
        def->getNumResults() != 1)
      return failure();

    auto defResTy = dyn_cast<ShapedType>(def->getResult(0).getType());
    if (!defResTy || !defResTy.hasRank() || defResTy.getRank() != 0)
      return failure();

    SmallVector<Value> scalarOperands;
    scalarOperands.reserve(def->getNumOperands());
    for (Value operand : def->getOperands()) {
      auto scalarOr = materializeScalarLikeValueAtIndexImpl(
          b, loc, operand, index, stack, depth + 1);
      if (failed(scalarOr))
        return failure();
      scalarOperands.push_back(*scalarOr);
    }

    auto clonedOr =
        clonePureOpWithScalarizedTypes(b, *def, ValueRange{scalarOperands});
    if (failed(clonedOr))
      return failure();
    if ((*clonedOr)->getNumResults() != 1)
      return failure();
    return (*clonedOr)->getResult(0);
  }

  // Higher-rank tensors are intentionally unsupported here.
  return failure();
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
  if (!destSubviewTy)
    return failure();

  auto resultTy =
      dyn_cast<RankedTensorType>(localizedGeneric.getResult(0).getType());
  if (!resultTy || destSubviewTy.getRank() != resultTy.getRank())
    return failure();

  if (destSubviewTy.getElementType() != resultTy.getElementType())
    return failure();

  if (static_cast<int64_t>(memrefInputs.size()) !=
      localizedGeneric.getNumDpsInputs())
    return failure();

  for (auto [inputOperand, memrefInput] :
       llvm::zip(localizedGeneric.getDpsInputOperands(), memrefInputs)) {
    auto inputTy = dyn_cast<RankedTensorType>(inputOperand->get().getType());
    auto memrefInputTy = dyn_cast<MemRefType>(memrefInput.getType());
    if (!inputTy || !memrefInputTy ||
        inputTy.getRank() != memrefInputTy.getRank())
      return failure();
    if (inputTy.getElementType() != memrefInputTy.getElementType())
      return failure();
  }

  auto memrefGeneric = linalg::GenericOp::create(
      b, localizedGeneric.getLoc(),
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

/// Try to lower a localized identity-map elementwise tensor producer directly
/// to a memref linalg.generic writing into `destSubview`.
///
/// This recognizes both forms produced around NSP localization:
///   1. Direct rank-N localized generic:
///        %tile = linalg.generic ... -> tensor<...>
///   2. Canonicalized rank-reduced generic re-expanded to the tile rank:
///        %collapsed_in = tensor.collapse_shape %all_slice_or_copy ...
///        %generic = linalg.generic ... -> tensor<16x512x4xf16>
///        %expanded = tensor.expand_shape %generic
///                    : tensor<16x512x4xf16> into tensor<1x16x512x4xf16>
///
/// Case (2) is important for rank-4 block_ptr tensors with a leading unit
/// dimension.  The fast-path preserves the rank-3 generic and collapses the
/// rank-4 destination/input subviews to matching rank-3 memref views, avoiding
/// alloc_tensor copies around the compute.
static LogicalResult tryRewriteLocalizedElementwiseGenericToMemref(
    OpBuilder &b, Location loc, mlir::hexagon::nsp::MaterializeTileOp tileOp,
    Value source, Value destSubview) {
  Value producer = stripTensorMaterializationWrappers(source);
  Value genericResult = producer;
  Value genericDestSubview = destSubview;

  if (auto expand = producer.getDefiningOp<tensor::ExpandShapeOp>()) {
    auto expandedTy = dyn_cast<RankedTensorType>(expand.getResult().getType());
    auto collapsedTy =
        dyn_cast<RankedTensorType>(expand->getOperand(0).getType());
    auto destSubviewTy = dyn_cast<MemRefType>(destSubview.getType());
    if (!expandedTy || !collapsedTy || !destSubviewTy)
      return failure();

    if (destSubviewTy.getRank() != expandedTy.getRank())
      return failure();
    if (destSubviewTy.getElementType() != expandedTy.getElementType())
      return failure();

    SmallVector<ReassociationIndices> reassociation =
        expand.getReassociationIndices();
    auto collapsedDestOr = buildCollapsedMemrefView(b, loc, destSubview,
                                                    reassociation, collapsedTy);
    if (failed(collapsedDestOr))
      return failure();

    genericDestSubview = *collapsedDestOr;
    genericResult = stripTensorMaterializationWrappers(expand->getOperand(0));
  }

  auto localizedGeneric = genericResult.getDefiningOp<linalg::GenericOp>();
  if (!isSimpleLocalizedElementwiseGeneric(localizedGeneric))
    return failure();

  Value oldInit = localizedGeneric.getDpsInitOperand(0)->get();
  Operation *oldGenericOp = localizedGeneric.getOperation();

  SmallVector<Value> memrefInputs;
  memrefInputs.reserve(localizedGeneric.getNumDpsInputs());

  for (OpOperand *in : localizedGeneric.getDpsInputOperands()) {
    auto subviewOr = materializeRankedInputAsSubview(b, loc, in->get());
    if (failed(subviewOr))
      return failure();
    memrefInputs.push_back(*subviewOr);
  }

  auto newGenericOr = createMemrefGenericAtCurrentInsertionPoint(
      b, localizedGeneric, memrefInputs, genericDestSubview);
  if (failed(newGenericOr))
    return failure();

  removeNSPLocalizedAttrs(localizedGeneric);
  return finalizeSuccessfulTensorToMemrefRewrite(tileOp, oldGenericOp, oldInit);
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

  return memref::SubViewOp::create(b, loc, subviewTy, baseMemref, offsets,
                                   sizes, strides)
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
  auto newFor = scf::ForOp::create(b, loc, forOp.getLowerBound(),
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

      auto newRead = vector::TransferReadOp::create(
          nb, tr.getLoc(), tr.getVectorType(), newChunkMemrefs[idx], indices,
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

      vector::TransferWriteOp::create(
          nb, tw.getLoc(), map.lookupOrDefault(tw.getVector()), outChunkMemref,
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

  scf::YieldOp::create(nb, loc);
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
    Value scalarLikeBase;
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
                                          chunkSizes[1], Value()});
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
                                          Value()});
        continue;
      }
    }

    // Scalar-broadcast case: accept a rank-0 tensor base, including a short
    // pure arith/math chain on top of the original scalar slice.
    Value scalarBase = stripTensorCastWrappers(tr.getBase());
    auto scalarBaseTy = dyn_cast<RankedTensorType>(scalarBase.getType());
    if (!scalarBaseTy || scalarBaseTy.getRank() != 0)
      return failure();

    // Current vectorized scalar-broadcast form reads a scalar tensor with no
    // explicit indices and broadcasts it via permutation_map = () -> (0).
    if (!tr.getIndices().empty())
      return failure();
    if (tr.getMask())
      return failure();

    readSpecs.push_back(InputReadSpec{InputReadKind::Rank1Scalar, Value(),
                                      OpFoldResult(), OpFoldResult(),
                                      scalarBase});
  }

  SmallVector<Value> inputTileMemrefs;
  inputTileMemrefs.reserve(readSpecs.size());
  for (const InputReadSpec &spec : readSpecs) {
    FailureOr<Value> memrefOr = failure();
    if (spec.kind == InputReadKind::Rank2Chunk)
      memrefOr = materializeRank2InputAsSubview(b, loc, spec.sourceTensor);

    // Rank1Scalar is materialized directly as a scalar during cloning of the
    // inner loop body, so no memref tile is needed up-front.
    if (spec.kind == InputReadKind::Rank1Scalar) {
      inputTileMemrefs.push_back(Value());
      continue;
    }
    if (failed(memrefOr))
      return failure();
    inputTileMemrefs.push_back(*memrefOr);
  }

  auto newOuter =
      scf::ForOp::create(b, loc, outerFor.getLowerBound(),
                         outerFor.getUpperBound(), outerFor.getStep());
  Block *newOuterBody = newOuter.getBody();
  newOuterBody->getOperations().clear();

  IRMapping outerMap;
  outerMap.map(outerFor.getInductionVar(), newOuter.getInductionVar());
  OpBuilder ob = OpBuilder::atBlockEnd(newOuterBody);

  auto newInner =
      scf::ForOp::create(ob, loc, innerFor.getLowerBound(),
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
        auto scalarOr = materializeScalarFromRank1TensorAtIndex(
            ib, tr.getLoc(), spec.scalarLikeBase, newOuter.getInductionVar());
        if (failed(scalarOr))
          return failure();

        auto broadcast = vector::BroadcastOp::create(
            ib, tr.getLoc(), tr.getVectorType(), *scalarOr);
        map.map(tr.getResult(), broadcast.getResult());

      } else {
        SmallVector<Value> indices;
        indices.reserve(tr.getIndices().size());
        for (Value iv : tr.getIndices())
          indices.push_back(map.lookupOrDefault(iv));

        auto newRead = vector::TransferReadOp::create(
            ib, tr.getLoc(), tr.getVectorType(), newChunkMemrefs[idx], indices,
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

      vector::TransferWriteOp::create(
          ib, tw.getLoc(), map.lookupOrDefault(tw.getVector()), outChunkMemref,
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

  scf::YieldOp::create(ib, loc);
  scf::YieldOp::create(ob, loc);
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
  if (!isSupportedIdentityElementwiseRank(rank))
    return failure();

  int64_t splitAxis = tileOp.getSplitAxis();
  if (splitAxis < 0 || splitAxis >= rank)
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
  Value procIdx = mlir::shard::ProcessLinearIndexOp::create(b, loc, grid);
  Value tileExtent =
      arith::ConstantIndexOp::create(b, loc, tileShape[splitAxis]);
  Value tileOffset = arith::MulIOp::create(b, loc, procIdx, tileExtent);

  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;

  offsets.reserve(rank);
  sizes.reserve(rank);
  strides.reserve(rank);

  for (int64_t d = 0; d < rank; ++d) {
    offsets.push_back(d == splitAxis ? OpFoldResult(tileOffset)
                                     : OpFoldResult(b.getIndexAttr(0)));
    sizes.push_back(b.getIndexAttr(tileShape[d]));
    strides.push_back(b.getIndexAttr(1));
  }

  auto subview =
      memref::SubViewOp::create(b, loc, dest, offsets, sizes, strides);
  Value destSubview = subview.getResult();

  // Pattern 1: localized elementwise linalg.generic.
  //
  // Accept both the direct localized generic and the canonicalized
  // collapse/generic/expand form produced for rank-4 tiles with a leading unit
  // dimension.  This is the path expected to handle GELU block_ptr without
  // alloc_tensor copies around the local compute.
  if (succeeded(tryRewriteLocalizedElementwiseGenericToMemref(
          b, loc, tileOp, source, destSubview)))
    return success();

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
    // If rank-2 matching fails, we still fall back to the logic below.
    if (succeeded(
            tryRewriteScfForTileToMemref(b, loc, tileLoop, destSubview))) {
      return finalizeSuccessfulTensorToMemrefRewrite(tileOp, oldLoopOp,
                                                     initArg);
    }
  }

  // Fallback: keep the original tensor path.
  bufferization::MaterializeInDestinationOp::create(
      b, loc, TypeRange{}, source, destSubview,
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
