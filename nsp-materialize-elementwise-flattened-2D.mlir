// RUN: linalg-hexagon-opt %s -nsp-materialize | FileCheck %s

// This test validates the rank-2 identity-map elementwise materialization path
// after the NSP grid was changed from a legacy 1-D grid to a 2-D grid.
//
// The input IR models the hand-off produced by NSPLocalizePass:
//   - the NSP grid is 2-D: 16 cores x 4 threads;
//   - the elementwise path still flattens all grid axes for this test;
//   - therefore, tensor<2048x512xf32> is split into 64 local tiles;
//   - each local tile has shape tensor<32x512xf32>;
//   - nsp.materialize_tile records that the local tile must be written into
//     the global destination memref<2048x512xf32>.
//
// Expected NSPMaterialize behavior:
//   - compute the flattened participant index directly from the Hexagon ABI:
//       linearIdx = cid * ntpc + tid
//   - create a destination memref.subview:
//       %out[linearIdx * 32, 0] [32, 512] [1, 1]
//   - recover memref views for the two tensor tile inputs;
//   - rebuild the elementwise linalg.generic with memref semantics;
//   - erase the temporary nsp.materialize_tile op;
//   - avoid the fallback bufferization.materialize_in_destination path.

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  shard.grid @nsp(shape = 16x4)

  func.func @materialize_rank2_elementwise(%a: memref<2048x512xf32>,
                                           %b: memref<2048x512xf32>,
                                           %out: memref<2048x512xf32>,
                                           %ntpc: i32,
                                           %num_cores: i32,
                                           %reserved0: i32,
                                           %tid: i32,
                                           %cid: i32,
                                           %reserved1: i32) {
    %c0 = arith.constant 0 : index

    %a_tensor = bufferization.to_tensor %a
      : memref<2048x512xf32> to tensor<2048x512xf32>
    %b_tensor = bufferization.to_tensor %b
      : memref<2048x512xf32> to tensor<2048x512xf32>

    %a_tile = tensor.extract_slice %a_tensor[%c0, %c0] [32, 512] [1, 1]
      : tensor<2048x512xf32> to tensor<32x512xf32>
    %b_tile = tensor.extract_slice %b_tensor[%c0, %c0] [32, 512] [1, 1]
      : tensor<2048x512xf32> to tensor<32x512xf32>

    %init = tensor.empty() : tensor<32x512xf32>

    %tile = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"],
      nsp.localized
    } ins(%a_tile, %b_tile : tensor<32x512xf32>, tensor<32x512xf32>)
      outs(%init : tensor<32x512xf32>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    } -> tensor<32x512xf32>

    nsp.materialize_tile %tile : tensor<32x512xf32> into %out : memref<2048x512xf32>
      grid @nsp
      split_axis = 0
      tile_shape = [32, 512]

    return
  }
}

// CHECK: shard.grid @nsp
// CHECK-SAME: shape = 16x4

// CHECK-LABEL: func.func @materialize_rank2_elementwise

// CHECK: %[[PID:.*]] = shard.process_linear_index on @nsp : index

// The elementwise flattened 2-D path writes a 32x512 tile into the global
// destination.
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK: %[[ROW_OFFSET:.*]] = arith.muli %{{.*}}, %[[C32]] : index

// The final destination becomes a per-participant rank-2 subview.
// CHECK: %[[DST_VIEW:.*]] = memref.subview %{{.*}}[%[[ROW_OFFSET]], 0] [32, 512] [1, 1]

// The localized generic should be rewritten to memref form.
// CHECK: linalg.generic
// CHECK-SAME: outs(%[[DST_VIEW]]

// The temporary NSP hand-off must be consumed.
// CHECK-NOT: nsp.materialize_tile
