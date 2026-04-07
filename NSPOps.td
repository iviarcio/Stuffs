//===- NSPOps.td - NSP Dialect operation definitions -------*- tablegen -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Private NSP operations used as a contract between localization and
// materialization passes.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_NSP_IR_NSP_OPS_TD
#define HEXAGON_DIALECT_NSP_IR_NSP_OPS_TD

include "hexagon/Dialect/NSP/IR/NSPDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

def NSP_MaterializeTileOp : NSP_Op<
    "materialize_tile",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]
> {
  let summary = "Anchor the later materialization of a shard-local tensor tile";
  let description = [{
    `nsp.materialize_tile` is a private, declarative contract between
    `NSPLocalizePass` and `NSPMaterializePass`.

    The operation states that `source` already represents the tensor tile local
    to the current NSP participant and must later be written into `dest`, a
    global destination memref. The exact destination slice is computed from the
    `grid`, `split_axis`, and `tile_shape` attributes by `NSPMaterializePass`.

    Keeping this as a dedicated operation avoids leaving the original global
    `linalg.generic` producer in the IR, which in turn prevents downstream
    tiling/vectorization from accidentally matching the wrong computation.
  }];

  let arguments = (ins
    AnyRankedTensor:$source,
    AnyMemRef:$dest,
    FlatSymbolRefAttr:$grid,
    I64Attr:$split_axis,
    I64ArrayAttr:$tile_shape
  );

  let assemblyFormat = [{
    $source `:` type($source) `into` $dest `:` type($dest)
    `grid` $grid
    `split_axis` `=` $split_axis
    `tile_shape` `=` $tile_shape
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::RankedTensorType getSourceType() {
      return ::llvm::cast<::mlir::RankedTensorType>(getSource().getType());
    }

    ::mlir::MemRefType getDestType() {
      return ::llvm::cast<::mlir::MemRefType>(getDest().getType());
    }

    void getEffects(
        ::llvm::SmallVectorImpl<
            ::mlir::MemoryEffects::EffectInstance> &effects);
  }];

  let hasVerifier = 1;
}

#endif // HEXAGON_DIALECT_NSP_IR_NSP_OPS_TD
