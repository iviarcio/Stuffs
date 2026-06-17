// RUN: linalg-hexagon-opt %s -nsp-shard-planner | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @vadd(%a: tensor<2048x512xf32>,
                  %b: tensor<2048x512xf32>) -> tensor<2048x512xf32> {
    %init = tensor.empty() : tensor<2048x512xf32>
    %0 = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b : tensor<2048x512xf32>, tensor<2048x512xf32>)
      outs(%init : tensor<2048x512xf32>) {
    ^bb0(%x: f32, %y: f32, %out: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    } -> tensor<2048x512xf32>

    return %0 : tensor<2048x512xf32>
  }
}

// CHECK: shard.grid @nsp
// CHECK: shard.sharding
// CHECK: shard.shard
// CHECK: linalg.generic
