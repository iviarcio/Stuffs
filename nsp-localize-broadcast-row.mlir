// RUN: linalg-hexagon-opt %s -nsp-shard-planner -nsp-localize | FileCheck %s

// This test validates the rank-2 row-broadcast localization path.
// Input:
//   C[i, j] = A[i, j] + row[i]
// where:
//   A   : tensor<2048x512xf32>
//   row : tensor<2048xf32>
//   C   : tensor<2048x512xf32>
// Expected NSP localization behavior:
//   - split the rank-2 output along axis 0;
//   - local output tile is tensor<128x512xf32>, because 2048 / 16 = 128;
//   - slice the rank-2 input A into tensor<128x512xf32>;
//   - slice the row-broadcast input row into tensor<128xf32>;
//   - rebuild the linalg.generic over local tensors;
//   - emit nsp.materialize_tile for the local rank-2 result.

#map_2d = affine_map<(d0, d1) -> (d0, d1)>
#map_row = affine_map<(d0, d1) -> (d0)>

module {
  func.func @broadcast_row(%a: tensor<2048x512xf32>,
                           %row: tensor<2048xf32>,
                           %out: memref<2048x512xf32>) {
    %init = tensor.empty() : tensor<2048x512xf32>

    %0 = linalg.generic {
      indexing_maps = [#map_2d, #map_row, #map_2d],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %row : tensor<2048x512xf32>, tensor<2048xf32>)
      outs(%init : tensor<2048x512xf32>) {
    ^bb0(%x: f32, %r: f32, %unused: f32):
      %sum = arith.addf %x, %r : f32
      linalg.yield %sum : f32
    } -> tensor<2048x512xf32>

    bufferization.materialize_in_destination
      %0 in restrict writable %out
        : (tensor<2048x512xf32>, memref<2048x512xf32>) -> ()

    return
  }
}

// CHECK: shard.grid @nsp
// CHECK-SAME: shape = 16

// CHECK-LABEL: func.func @broadcast_row

// The rank-2 matrix input is sliced into the rank-2 local tile.
// CHECK-DAG: %[[A_TILE:.*]] = shard.all_slice
// CHECK-DAG: tensor<128x512xf32>

// The row-broadcast input is also sliced, but only along the row dimension.
// CHECK-DAG: %[[ROW_TILE:.*]] = shard.all_slice
// CHECK-DAG: tensor<128xf32>

// The localized init has the rank-2 local output shape.
// CHECK: %[[LOCAL_INIT:.*]] = tensor.empty() : tensor<128x512xf32>

// The localized generic consumes the rank-2 tile and the rank-1 row tile.
// CHECK: %[[LOCAL_RESULT:.*]] = linalg.generic
// CHECK: ins(%[[A_TILE]], %[[ROW_TILE]]
// CHECK: outs(%[[LOCAL_INIT]]
// CHECK: arith.addf
// CHECK: linalg.yield
// CHECK: -> tensor<128x512xf32>

// The localized result is handed off to NSPMaterializePass.
// CHECK: nsp.materialize_tile %[[LOCAL_RESULT]] : tensor<128x512xf32> into %{{.*}} : memref<2048x512xf32>
// CHECK-SAME: grid @nsp
// CHECK-SAME: split_axis = 0
// CHECK-SAME: tile_shape = [128, 512]
