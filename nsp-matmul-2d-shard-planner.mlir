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
    // CHECK: shard.grid @nsp(shape = 16x4)
    // CHECK-LABEL: func.func @matmul_2d_grid_plan

    // Match all descriptors by their split_axes first, then check which operand uses each one.
    // CHECK-DAG: %[[SH_C:[A-Za-z0-9_]+]] = shard.sharding @nsp split_axes = {{\[\[0\], \[1\]\]}} : !shard.sharding
    // CHECK-DAG: %[[SH_B:[A-Za-z0-9_]+]] = shard.sharding @nsp split_axes = {{\[\[\], \[1\]\]}} : !shard.sharding
    // CHECK-DAG: %[[SH_A:[A-Za-z0-9_]+]] = shard.sharding @nsp split_axes = {{\[\[0\], \[\]\]}} : !shard.sharding

    // CHECK: %[[A_SHARDED:[A-Za-z0-9_]+]] = shard.shard %arg0 to %[[SH_A]] annotate_for_users : tensor<128x32xf32>
    // CHECK: %[[B_SHARDED:[A-Za-z0-9_]+]] = shard.shard %arg1 to %[[SH_B]] annotate_for_users : tensor<32x64xf32>
    // CHECK: %[[C_SHARDED:[A-Za-z0-9_]+]] = shard.shard %arg2 to %[[SH_C]] annotate_for_users : tensor<128x64xf32>

    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[A_SHARDED]], %[[B_SHARDED]]
    // CHECK-SAME: outs(%[[C_SHARDED]]

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
