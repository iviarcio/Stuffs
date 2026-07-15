// RUN: linalg-hexagon-opt %s -nsp-shard-planner | FileCheck %s

// This test validates that the NSP shard planner recognizes a matmul-like
// linalg.generic and maps the two parallel output iterators to distinct axes
// of the 2-D NSP grid instead of flattening all grid axes onto one dimension.

#map_a = affine_map<(i, j, k) -> (i, k)>
#map_b = affine_map<(i, j, k) -> (k, j)>
#map_c = affine_map<(i, j, k) -> (i, j)>

module {
  func.func @matmul_2d_grid_plan(
      %a: tensor<128x32xf32>,
      %b: tensor<32x64xf32>,
      %init: tensor<128x64xf32>) -> tensor<128x64xf32> {

    %0 = linalg.generic {
        indexing_maps = [#map_a, #map_b, #map_c],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%a, %b : tensor<128x32xf32>, tensor<32x64xf32>)
        outs(%init : tensor<128x64xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
        %mul = arith.mulf %lhs, %rhs : f32
        %add = arith.addf %acc, %mul : f32
        linalg.yield %add : f32
    } -> tensor<128x64xf32>

    return %0 : tensor<128x64xf32>
  }
}

// CHECK: shard.grid @nsp
// CHECK-SAME: shape = 16x4

// CHECK-LABEL: func.func @materialize_matmul_2d_offsets

// The materializer emits a symbolic participant index. ABI lowering is left to
// ShardToLLVM.
// CHECK: %[[PID:.*]] = shard.process_linear_index on @nsp : index

// For a 16x4 grid, axis 0 is recovered with div by 4 and axis 1 with rem by 4.
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[CORE_IDX:.*]] = arith.divui %[[PID]], %[[C4]] : index
// CHECK-DAG: %[[THREAD_IDX:.*]] = arith.remui %[[PID]], %[[C4]] : index

// CHECK-DAG: %[[TILE_M:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[TILE_N:.*]] = arith.constant 64 : index

// CHECK: %[[ROW_OFFSET:.*]] = arith.muli %[[CORE_IDX]], %[[TILE_M]] : index
// CHECK: %[[COL_OFFSET:.*]] = arith.muli %[[THREAD_IDX]], %[[TILE_N]] : index

// CHECK: %[[DST_VIEW:.*]] = memref.subview %{{.*}}[%[[ROW_OFFSET]], %[[COL_OFFSET]]] [128, 64] [1, 1]

// The temporary hand-off must be consumed.
// CHECK-NOT: nsp.materialize_tile

