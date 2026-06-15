//===- NSPDialect.td - NSP Dialect -------------------------*- tablegen -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// NSP dialect used as an IR contract between NSPLocalizePass and
// NSPMaterializePass. This dialect is intentionally local to hexagon-mlir
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_NSP_IR_NSP_DIALECT_TD
#define HEXAGON_DIALECT_NSP_IR_NSP_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonTypeConstraints.td"

//===----------------------------------------------------------------------===//
// NSP Dialect
//===----------------------------------------------------------------------===//

def NSP_Dialect : Dialect {
  let name = "nsp";
  let cppNamespace = "::mlir::hexagon::nsp";

  let description = [{
    NSP dialect used only inside the Hexagon NSP pipeline.

    NSP operations are temporary marker operations. They are not target
    operations and are not expected to survive final lowering. Instead, they
    make hand-off points between NSP-specific passes explicit in the IR.
    For example, after NSPLocalizePass, a global tensor computation such as:

    ```mlir
    %result = linalg.generic ... -> tensor<2048x512xf32>
    ```

    may be replaced by shard-local tensor computation plus an NSP marker:

    ```mlir
    %tile = linalg.generic ... -> tensor<64x512xf32>

    nsp.materialize_tile %tile : tensor<64x512xf32>
      into %dest : memref<2048x512xf32>
      grid @nsp
      split_axis = 0
      tile_shape = [64, 512]
    ```

    This lets downstream tiling and vectorization operate on the local tensor
    computation, while NSPMaterializePass later turns the marker into explicit
    destination-buffer writes, such as memref.subview-based stores.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::memref::MemRefDialect",
    "mlir::shard::ShardDialect"
  ];
}

class NSP_Op<string mnemonic, list<Trait> traits = []> :
    Op<NSP_Dialect, mnemonic, traits> {
}

#endif // HEXAGON_DIALECT_NSP_IR_NSP_DIALECT_TD
