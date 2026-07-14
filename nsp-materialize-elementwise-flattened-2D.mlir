// RUN: linalg-hexagon-opt %s -nsp-materialize | FileCheck %s

// This test validates the rank-2 identity-map elementwise materialization path
// after the NSP grid was changed from a legacy 1-D grid to a 2-D grid.
//
// The input IR models the hand-off produced by NSPLocalizePass:
//
//   - the NSP grid is 2-D: 16 cores x 4 threads;
//   - the elementwise path still flattens all grid axes for this test;
//   - therefore, tensor<2048x512xf32> is split into 64 local tiles;
//   - each local tile has shape tensor<32x512xf32>;
//   - nsp.materialize_tile records that the local tile must be written into
//     the global destination memref<2048x512xf32>.
//
// Expected NSPMaterialize behavior:
//
//   - emit shard.process_linear_index for the flattened per-participant index;
//   - create a destination memref.subview:
//
//       %out[linearIdx * 32, 0] [32, 512] [1, 1]
//
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

// nsp-materialize emits shard.process_linear_index for the flattened
// per-participant index.
// CHECK: %[[LINEAR_IDX:.*]] = shard.process_linear_index on @nsp : index

// The destination offset is linearIdx * tile_shape[0], i.e. linearIdx * 32.
// CHECK: %[[C32:.*]] = arith.constant 32 : index
// CHECK: %[[ROW_OFFSET:.*]] = arith.muli %[[LINEAR_IDX]], %[[C32]] : index

// The final destination becomes a per-participant rank-2 subview.
// CHECK: %[[DST_VIEW:.*]] = memref.subview %{{.*}}[%[[ROW_OFFSET]], 0] [32, 512] [1, 1]

// The tensor tile inputs are recovered as memref subviews over the original
// input buffers. These are local 32x512 views in the flattened 2-D grid path.
// CHECK: %[[A_VIEW:.*]] = memref.subview %{{.*}}[%c0, %c0] [32, 512] [1, 1]
// CHECK: %[[B_VIEW:.*]] = memref.subview %{{.*}}[%c0, %c0] [32, 512] [1, 1]

// The localized tensor linalg.generic is rewritten as a memref linalg.generic
// consuming the input subviews and writing directly into the destination subview.
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[A_VIEW]], %[[B_VIEW]]
// CHECK-SAME: outs(%[[DST_VIEW]]
// CHECK: arith.addf
// CHECK: linalg.yield

// The direct memref rewrite should succeed, so the fallback path must not be
// used and the temporary hand-off op must disappear.
// CHECK-NOT: bufferization.materialize_in_destination
// CHECK-NOT: nsp.materialize_tile

// CHECK: return
