//===- NSPOps.td - NSP Dialect operation definitions -------*- tablegen -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Operations in the NSP dialect. Currently contains only materialize_tile,
// the hand-off contract between NSPLocalizePass and NSPMaterializePass.
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
    `nsp.materialize_tile` is a declarative contract between `NSPLocalizePass`
    (producer) and `NSPMaterializePass` (consumer).

    Semantics
    ---------
    The op states that `source` is a shard-local tensor tile already computed by
    participant `p`, and that `NSPMaterializePass` must later write it into
    `dest` (a global destination memref) at an offset determined by `p`.

    Along `split_axis`, the destination slice for participant `p` is:

      dest[p * tile_shape[split_axis] :
           (p + 1) * tile_shape[split_axis], ...]

    All other dimensions use offset 0 and size `tile_shape[d]`. In the common
    case, those dimensions cover the full extent of the destination tensor.

    Why this op exists
    ------------------
    After `NSPLocalizePass` rewrites a global `linalg.generic` into shard-local
    tensor compute, the original global producer must be erased so that
    downstream tiling/vectorization does not match the wrong computation.

    However, the final destination-buffer write cannot be materialized yet:
    tiling/vectorization may still run between localization and materialization.
    This op therefore holds the hand-off point in the IR. It carries the
    information needed to materialize the tile later, without requiring either a
    live global tensor producer or premature memref lowering.

    The op is modeled conservatively as having memory effects, including a
    write effect, so that it is not eliminated before `NSPMaterializePass` runs
    and ordering relative to surrounding memory operations is preserved.

    Example
    -------
    If a global `tensor<2048x512xf32>` result is split along dimension 0 across
    a 1-D NSP grid with 32 participants, each participant computes a local
    `tensor<64x512xf32>` tile:

    ```mlir
    shard.grid @nsp(shape = 32)

    // Participant p writes %tile into:
    //   dest[p * 64 : (p + 1) * 64, 0 : 512]
    nsp.materialize_tile %tile : tensor<64x512xf32>
      into %dest : memref<2048x512xf32>
      grid @nsp
      split_axis = 0
      tile_shape = [64, 512]
    ```

    `NSPMaterializePass` then lowers this marker schematically to:

    ```mlir
    %p = shard.process_linear_index ...
    %offset = arith.muli %p, %c64 : index
    %subview = memref.subview %dest[%offset, 0][64, 512][1, 1]
    // ... write %tile into %subview ...
    ```
  }];

  let arguments = (ins
    AnyRankedTensor: $source,
    AnymemRef:$dest,
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
  }]

  let extraClassDeclaration = [{
    ::mlir::RankedTensorType getSourceType() {
      return ::llvm::cast<::mlir::RankedTensorType>(getSource().getType());
    }

    ::mlir::MemRefType getDestType() {
      return ::llvm::cast<::mlir::MemRefType>(getDest().getType());
    }
  }];

  let hasVerifier = 1;
}

#endif // HEXAGON_DIALECT_NSP_IR_NSP_OPS_TD
