//===- NSPOps.cpp ---------------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Private NSP operations used by the Hexagon NSP pipeline.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/NSP/IR/NSPOps.h"

#include "mlir/Dialect/Shard/IR/ShardOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::hexagon::nsp;

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "hexagon/Dialect/NSP/IR/NSPOps.cpp.inc"

static LogicalResult verifyGridRef(Operation *op, FlatSymbolRefAttr gridAttr) {
  if (!gridAttr)
    return op->emitOpError() << "requires a non-null 'grid' symbol reference";

  auto module = op->getParentOfType<ModuleOp>();
  if (!module)
    return success();

  auto grid =
      SymbolTable::lookupNearestSymbolFrom<mlir::shard::GridOp>(op, gridAttr);
  if (!grid)
    return op->emitOpError()
           << "'grid' must reference a valid shard.grid symbol";

  return success();
}

void MaterializeTileOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // This op is a semantic anchor for a later materialization into `dest`.
  // Model it conservatively as:
  //   - reading the shard-local source tensor,
  //   - reading the destination memref,
  //   - writing the destination memref.
  //
  // This prevents canonicalization / DCE from erasing the op and its producer
  // chain before NSPMaterializePass runs.
  effects.emplace_back(MemoryEffects::Read::get(), getSource(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getDest(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), getDest(),
                       SideEffects::DefaultResource::get());
}

LogicalResult MaterializeTileOp::verify() {
  RankedTensorType sourceTy = getSourceType();
  MemRefType destTy = getDestType();

  if (!sourceTy)
    return emitOpError() << "expects 'source' to be a ranked tensor";
  if (!destTy)
    return emitOpError() << "expects 'dest' to be a memref";

  if (failed(verifyGridRef(*this, getGridAttr())))
    return failure();

  int64_t splitAxis = getSplitAxis();
  auto tileShapeAttr = getTileShape();

  if (!tileShapeAttr || tileShapeAttr.empty())
    return emitOpError() << "requires non-empty 'tile_shape'";

  int64_t sourceRank = sourceTy.getRank();
  int64_t destRank = destTy.getRank();
  int64_t tileRank = static_cast<int64_t>(tileShapeAttr.size());

  if (sourceRank != destRank)
    return emitOpError()
           << "requires source and destination to have the same rank";

  if (tileRank != sourceRank)
    return emitOpError()
           << "requires 'tile_shape' rank to match source/destination rank";

  if (splitAxis < 0 || splitAxis >= destRank)
    return emitOpError()
           << "requires 'split_axis' to be within destination rank";

  if (splitAxis >= sourceRank)
    return emitOpError()
           << "requires 'split_axis' to be within source rank";

  if (sourceTy.getElementType() != destTy.getElementType())
    return emitOpError()
           << "requires matching element types for source and destination";

  for (int64_t d = 0; d < tileRank; ++d) {
    int64_t tileDim = tileShapeAttr[d];
    if (tileDim <= 0)
      return emitOpError()
             << "requires all entries of 'tile_shape' to be positive, but got "
             << tileShapeAttr;

    int64_t srcDim = sourceTy.getDimSize(d);
    if (!ShapedType::isDynamic(srcDim) && srcDim != tileDim)
      return emitOpError()
             << "requires source dimension " << d
             << " to match 'tile_shape', but got source dim = " << srcDim
             << " and tile_shape dim = " << tileDim;

    int64_t dstDim = destTy.getDimSize(d);
    if (!ShapedType::isDynamic(dstDim) && tileDim > dstDim)
      return emitOpError()
             << "requires 'tile_shape' dimension " << d
             << " to fit within destination dimension, but got tile_shape dim = "
             << tileDim << " and destination dim = " << dstDim;
  }

  return success();
}
