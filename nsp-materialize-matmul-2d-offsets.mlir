// RUN: linalg-hexagon-opt %s -nsp-materialize | FileCheck %s

// This test validates that NSPMaterialize computes independent destination
// offsets for a 2-D matmul tile using symbolic shard.process_linear_index.
//
// For a 16x4 grid, the materializer derives:
//   row participant = linearIdx / 4
//   col participant = linearIdx % 4
//
// The final lowering from shard.process_linear_index to the Hexagon ABI is left
// to ShardToLLVM.

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

  // CHECK: %[[PID0:.*]] = shard.process_linear_index on @nsp : index
  // CHECK: %[[C4_0:.*]] = arith.constant 4 : index
  // CHECK: %[[AXIS0:.*]] = arith.divui %[[PID0]], %[[C4_0]] : index
  // CHECK: %[[C8:.*]] = arith.constant 8 : index
  // CHECK: %[[ROW_OFF:.*]] = arith.muli %[[AXIS0]], %[[C8]] : index

  // CHECK: %[[PID1:.*]] = shard.process_linear_index on @nsp : index
  // CHECK: %[[C4_1:.*]] = arith.constant 4 : index
  // CHECK: %[[AXIS1:.*]] = arith.remui %[[PID1]], %[[C4_1]] : index
  // CHECK: %[[C16:.*]] = arith.constant 16 : index
  // CHECK: %[[COL_OFF:.*]] = arith.muli %[[AXIS1]], %[[C16]] : index

  // CHECK: %[[DST:.*]] = memref.subview %arg1[%[[ROW_OFF]], %[[COL_OFF]]] [8, 16] [1, 1]
  // CHECK-SAME: memref<128x64xf32> to memref<8x16xf32

  // CHECK: bufferization.materialize_in_destination %arg0 in writable %[[DST]]
  // CHECK-SAME: (tensor<8x16xf32>, memref<8x16xf32

  // CHECK-NOT: nsp.materialize_tile

    nsp.materialize_tile %tile : tensor<8x16xf32> into %dest : memref<128x64xf32>
      grid @nsp split_axis = 0 tile_shape = [8, 16]
    return
  }
}
