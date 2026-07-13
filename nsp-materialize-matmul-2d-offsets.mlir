// RUN: linalg-hexagon-opt %s -nsp-materialize | FileCheck %s

// This test validates that NSPMaterialize computes independent destination
// offsets for a 2-D matmul tile: rows use core id and columns use thread id.

module {
  shard.grid @nsp(shape = 16x4)

  func.func @materialize_matmul_2d_offsets(
      %tile: tensor<8x16xf32>,
      %dest: memref<128x64xf32>,
      %ntpc: index,
      %num_cores: index,
      %reserved0: index,
      %tid: index,
      %cid: index,
      %reserved1: index) {
    // CHECK-LABEL: func.func @materialize_matmul_2d_offsets

    // CHECK-DAG: %[[C8:[A-Za-z0-9_]+]] = arith.constant 8 : index
    // CHECK-DAG: %[[C16:[A-Za-z0-9_]+]] = arith.constant 16 : index
    // CHECK-DAG: %[[ROW_OFF:[A-Za-z0-9_]+]] = arith.muli %arg6, %[[C8]] : index
    // CHECK-DAG: %[[COL_OFF:[A-Za-z0-9_]+]] = arith.muli %arg5, %[[C16]] : index

    // CHECK: memref.subview %arg1[%[[ROW_OFF]], %[[COL_OFF]]] [8, 16] [1, 1]
    // CHECK-NOT: nsp.materialize_tile

    nsp.materialize_tile %tile : tensor<8x16xf32> into %dest : memref<128x64xf32>
        grid @nsp split_axis = 0 tile_shape = [8, 16]
    return
  }
}
