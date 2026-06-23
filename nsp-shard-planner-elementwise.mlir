// RUN: linalg-hexagon-opt %s -nsp-shard-planner | FileCheck %s

// This test validates the basic elementwise planning path of
// NSPShardPlanner.
// Input:
//   A pure tensor linalg.generic computing:
//     C[i, j] = A[i, j] + B[i, j]
// Expected planning decision:
//   - create the default 1-D NSP grid @nsp;
//   - choose loop dimension d0 for sharding, because output dim 0 has extent
//     2048 and is evenly divisible by the default NSP count, 16;
//   - create one shared !shard.sharding descriptor with:
//       split_axes = [[0], []]
//     meaning:
//       tensor dim 0 is split over grid axis 0;
//       tensor dim 1 is replicated;
//   - wrap both input tensors and the output init tensor with shard.shard;
//   - rebuild linalg.generic so it consumes the sharded SSA values.

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
// CHECK-SAME: shape = 16

// CHECK-LABEL: func.func @vadd

// CHECK: %[[SHARDING:.*]] = shard.sharding @nsp{{.*}}split_axes = {{\[\[0\], \[\]\]}}

// CHECK-DAG: %[[A_SHARDED:.*]] = shard.shard %{{.*}} to %[[SHARDING]]{{.*}}tensor<2048x512xf32>
// CHECK-DAG: %[[B_SHARDED:.*]] = shard.shard %{{.*}} to %[[SHARDING]]{{.*}}tensor<2048x512xf32>
// CHECK-DAG: %[[INIT_SHARDED:.*]] = shard.shard %{{.*}} to %[[SHARDING]]{{.*}}tensor<2048x512xf32>

// CHECK: %[[RESULT:.*]] = linalg.generic
// CHECK-SAME: ins(%[[A_SHARDED]], %[[B_SHARDED]]
// CHECK-SAME: outs(%[[INIT_SHARDED]]

// CHECK: arith.addf
// CHECK: linalg.yield
// CHECK-NOT: shard.all_reduce
// CHECK: return %[[RESULT]]
