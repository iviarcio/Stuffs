// RUN: linalg-hexagon-opt %s \
// RUN:   --pass-pipeline="builtin.module(nsp-shard(nsp-count=4 canonicalize=false))" \
// RUN:   | FileCheck %s

// CHECK: shard.grid @nsp
// CHECK-SAME: shape = 4x1

// CHECK-LABEL: func.func @vadd_1d

// CHECK-DAG: shard.all_slice %arg0 on @nsp grid_axes = [0, 1] slice_axis = 0 : tensor<64xf32> -> tensor<16xf32>
// CHECK-DAG: shard.all_slice %arg1 on @nsp grid_axes = [0, 1] slice_axis = 0 : tensor<64xf32> -> tensor<16xf32>

// CHECK: linalg.generic
// CHECK-SAME: ins({{.*}} : tensor<16xf32>, tensor<16xf32>)
// CHECK-SAME: outs({{.*}} : tensor<16xf32>)

// Materialization should create a per-participant destination subview.
// CHECK: shard.process_linear_index on @nsp : index
// CHECK: memref.subview %arg2[{{.*}}] [16] [1]

// Since the function arguments are tensors, NSPMaterialize cannot recover
// input memref subviews for a direct memref linalg.generic rewrite. It should
// fall back to materialize_in_destination into the destination subview.
// CHECK: bufferization.materialize_in_destination

// CHECK-NOT: nsp.materialize_tile

#map = affine_map<(d0) -> (d0)>

func.func @vadd_1d(%arg0: tensor<64xf32>,
                   %arg1: tensor<64xf32>,
                   %dest: memref<64xf32>) {
  %init = tensor.empty() : tensor<64xf32>

  %0 = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<64xf32>, tensor<64xf32>)
    outs(%init : tensor<64xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %add = arith.addf %a, %b : f32
    linalg.yield %add : f32
  } -> tensor<64xf32>

  bufferization.materialize_in_destination %0 in writable %dest
    : (tensor<64xf32>, memref<64xf32>) -> ()

  return
}
