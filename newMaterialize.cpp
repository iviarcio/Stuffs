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
//   - only handles rank-1 direct store-by-tile materialization,
//   - expects static tile size,
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
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mlir/Dialect/Shard/IR/ShardDialect.h"
#include "mlir/Dialect/Shard/IR/ShardOps.h"

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

/// Build the memref local view corresponding to a tensor local slice.
/// Supported patterns:
///   - %t = shard.all_slice %src ...
///       where %src comes from bufferization.to_tensor %buf
///   - %t = tensor.extract_slice %src[%off][%size][1]
///       where %src comes from bufferization.to_tensor %buf
///
/// Returns a rank-1 memref.subview over the original memref source.
///
/// Important:
///   This helper creates any required arithmetic/subview IR at the builder's
///   current insertion point. Callers must ensure that point dominates the
///   future use sites.
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
/// Important:
///   This helper intentionally does not change the insertion point. The caller
///   must position the builder such that all memref inputs and outputs dominate
///   the new generic.
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
  generic->removeAttr("nsp.materialize_tile_size");
  generic->removeAttr("nsp.materialize_split_axis");
  generic->removeAttr("nsp.group_id");
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

  if (sourceTy.getRank() != 1 || destTy.getRank() != 1)
    return failure();

  int64_t splitAxis = tileOp.getSplitAxis();
  int64_t tileSize = tileOp.getTileSize();
  if (splitAxis != 0 || tileSize <= 0)
    return failure();

  // All fast-path IR must be created at the tile op insertion point so that
  // the generated values dominate the new memref generic.
  b.setInsertionPoint(tileOp);

  Value procIdx = b.create<mlir::shard::ProcessLinearIndexOp>(loc, grid);
  Value tileSizeVal = b.create<arith::ConstantIndexOp>(loc, tileSize);
  Value offset = b.create<arith::MulIOp>(loc, procIdx, tileSizeVal);

  SmallVector<OpFoldResult> offsets = {offset};
  SmallVector<OpFoldResult> sizes = {b.getIndexAttr(tileSize)};
  SmallVector<OpFoldResult> strides = {b.getIndexAttr(1)};

  auto subLayout = StridedLayoutAttr::get(destTy.getContext(),
                                          /*offset=*/ShapedType::kDynamic,
                                          /*strides=*/ArrayRef<int64_t>{1});
  auto subviewTy =
      MemRefType::get(ArrayRef<int64_t>{tileSize}, destTy.getElementType(),
                      subLayout, destTy.getMemorySpace());

  Value destSubview = b.create<memref::SubViewOp>(loc, subviewTy, dest, offsets,
                                                  sizes, strides);

  // Fast-path: convert the full generic to buffer semantics.
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

          // Erase the hand-off op first, since it is the last user of the old
          // tensor result.
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
                    mlir::memref::MemRefDialect, mlir::tensor::TensorDialect,
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
