// RUN: linalg-hexagon-opt %s -shard-to-llvm | FileCheck %s

// Test: shard.all_slice with grid_axes = [0, 1] lowers to
// tensor.extract_slice using the flattened participant index.
//
// For the Hexagon ABI tail:
//
//   [..., ntpc, num_cores, reserved0, tid, cid, reserved1]
//
// this means:
//
//   linearIdx = cid * ntpc + tid
//   offset    = linearIdx * localDimSize
//
// Input:
//   tensor<4096xf32>, grid = 4x2, grid_axes = [0, 1], slice_axis = 0
//
// Output:
//   tensor<512xf32>, offset = (cid * ntpc + tid) * 512

shard.grid @nsp(shape = 4x2)

func.func @kernel_flattened(%arg0: memref<?xf32>, %arg1: memref<?xf32>,
                            %arg2: i32, %arg3: i32, %arg4: i32,
                            %arg5: i32, %arg6: i32, %arg7: i32)
    -> tensor<512xf32> {
  %rcast = memref.reinterpret_cast %arg0 to
      offset: [0], sizes: [4096], strides: [1]
      : memref<?xf32> to memref<4096xf32, strided<[1]>>

  %input = bufferization.to_tensor %rcast restrict
      : memref<4096xf32, strided<[1]>> to tensor<4096xf32>

  %local = shard.all_slice %input on @nsp grid_axes = [0, 1] slice_axis = 0
      : tensor<4096xf32> -> tensor<512xf32>

  return %local : tensor<512xf32>
}

// CHECK-LABEL: func.func @kernel_flattened(
// CHECK-SAME: %[[ARG0:.*]]: memref<?xf32>
// CHECK-SAME: %{{.*}}: memref<?xf32>
// CHECK-SAME: %[[NTPC_ARG:.*]]: i32
// CHECK-SAME: %{{.*}}: i32
// CHECK-SAME: %{{.*}}: i32
// CHECK-SAME: %[[TID_ARG:.*]]: i32
// CHECK-SAME: %[[CID_ARG:.*]]: i32
// CHECK-SAME: %{{.*}}: i32

// CHECK-DAG: %[[C512:.*]] = arith.constant 512 : index

// CHECK: %[[BASE:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: [0], sizes: [4096], strides: [1]
// CHECK: %[[TENSOR:.*]] = bufferization.to_tensor %[[BASE]] restrict

// grid_axes = [0, 1] means flattened participant index:
//   cid * ntpc + tid
// CHECK-DAG: %[[CID:.*]] = arith.index_cast %[[CID_ARG]] : i32 to index
// CHECK-DAG: %[[TID:.*]] = arith.index_cast %[[TID_ARG]] : i32 to index
// CHECK-DAG: %[[NTPC:.*]] = arith.index_cast %[[NTPC_ARG]] : i32 to index
// CHECK: %[[MUL:.*]] = arith.muli %[[CID]], %[[NTPC]] : index
// CHECK: %[[LINEAR_IDX:.*]] = arith.addi %[[MUL]], %[[TID]] : index
// CHECK: %[[OFFSET:.*]] = arith.muli %[[LINEAR_IDX]], %[[C512]] : index

// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[TENSOR]][%[[OFFSET]]] [512] [1] : tensor<4096xf32> to tensor<512xf32>
// CHECK: return %[[SLICE]] : tensor<512xf32>

// CHECK-NOT: shard.all_slice
// CHECK-NOT: shard.grid
