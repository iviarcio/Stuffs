// RUN: linalg-hexagon-opt %s -nsp-localize | FileCheck %s

// This test validates that NSPLocalize consumes the 2-D matmul sharding plan
// produced by NSPShardPlanner and materializes local tensor slices per grid
// axis instead of flattening cores and threads onto a single tensor dimension.

#map_a = affine_map<(i, j, k) -> (i, k)>
#map_b = affine_map<(i, j, k) -> (k, j)>
#map_c = affine_map<(i, j, k) -> (i, j)>

module {
  shard.grid @nsp(shape = 16x4)

  func.func @localize_matmul_2d(
      %a: tensor<128x32xf32>,
      %b: tensor<32x64xf32>,
      %init: tensor<128x64xf32>,
      %dest: memref<128x64xf32>) {
    // CHECK-LABEL: func.func @localize_matmul_2d

    %sh_a = shard.sharding @nsp split_axes = [[0], []] : !shard.sharding
    %sh_b = shard.sharding @nsp split_axes = [[], [1]] : !shard.sharding
    %sh_c = shard.sharding @nsp split_axes = [[0], [1]] : !shard.sharding

    %a_s = shard.shard %a to %sh_a annotate_for_users : tensor<128x32xf32>
    %b_s = shard.shard %b to %sh_b annotate_for_users : tensor<32x64xf32>
    %init_s = shard.shard %init to %sh_c annotate_for_users : tensor<128x64xf32>

    // A(i, k): rows are split by grid axis 0; k remains local/replicated.
    // CHECK-DAG: %[[A_LOCAL:[A-Za-z0-9_]+]] = shard.all_slice {{.*}}grid_axes = [0]{{.*}}slice_axis = 0{{.*}}tensor<8x32xf32>

    // B(k, j): columns are split by grid axis 1; k remains local/replicated.
    // CHECK-DAG: %[[B_LOCAL:[A-Za-z0-9_]+]] = shard.all_slice {{.*}}grid_axes = [1]{{.*}}slice_axis = 1{{.*}}tensor<32x16xf32>

    // C(i, j): the init/output tile is sliced first by rows, then columns.
    // CHECK-DAG: %[[C_ROW_LOCAL:[A-Za-z0-9_]+]] = shard.all_slice {{.*}}grid_axes = [0]{{.*}}slice_axis = 0{{.*}}tensor<8x64xf32>
    // CHECK-DAG: %[[C_LOCAL:[A-Za-z0-9_]+]] = shard.all_slice %[[C_ROW_LOCAL]] {{.*}}grid_axes = [1]{{.*}}slice_axis = 1{{.*}}tensor<8x16xf32>

    // CHECK: %[[LOCAL_RESULT:[A-Za-z0-9_]+]] = linalg.generic
    // CHECK-SAME: ins(%[[A_LOCAL]], %[[B_LOCAL]] : tensor<8x32xf32>, tensor<32x16xf32>)
    // CHECK-SAME: outs(%[[C_LOCAL]] : tensor<8x16xf32>)

    // CHECK: nsp.materialize_tile %[[LOCAL_RESULT]] : tensor<8x16xf32> into %arg3 : memref<128x64xf32>
    // CHECK-SAME: grid @nsp
    // CHECK-SAME: split_axis = 0
    // CHECK-SAME: tile_shape = [8, 16]

    %0 = linalg.generic {
        indexing_maps = [#map_a, #map_b, #map_c],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%a_s, %b_s : tensor<128x32xf32>, tensor<32x64xf32>)
        outs(%init_s : tensor<128x64xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
        %mul = arith.mulf %lhs, %rhs : f32
        %add = arith.addf %acc, %mul : f32
        linalg.yield %add : f32
    } -> tensor<128x64xf32>

    "bufferization.materialize_in_destination"(%0, %dest) <{writable}> :
        (tensor<128x64xf32>, memref<128x64xf32>) -> ()
    return
  }
}
