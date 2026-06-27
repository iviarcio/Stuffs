// RUN: linalg-hexagon-opt %s -nsp-shard-planner -nsp-localize | FileCheck %s

// This test validates localization of an scf.for with tensor iter_args.
// Input shape:
//   tensor<2048x512xf32>
// NSP grid:
//   @nsp(shape = 16)
// Expected local shape:
//   tensor<128x512xf32>
//
// This is not the DOALL loop-distribution path. The scf.for has tensor
// iter_args and tensor results, so it must be localized by cloning the loop
// with local tensor iter_args when a downstream localized consumer requests
// a local version of the loop result.
//
// High-level input:
//   %loop = scf.for ... iter_args(%acc = %init)
//       -> tensor<2048x512xf32> {
//     %next = linalg.generic ins(%acc, %a) outs(%tmp)
//     scf.yield %next
//   }
//   %result = linalg.generic ins(%loop, %b) outs(%final_init)
//   bufferization.materialize_in_destination %result in %out
//
// Expected output:
//   - external tensor values used by the localized loop are sliced;
//   - a new scf.for is created with tensor<128x512xf32> iter_args/results;
//   - the localized scf.for is marked with nsp.localized_loop;
//   - the downstream consumer is localized over tensor<128x512xf32>;
//   - the final local tile is handed off through nsp.materialize_tile;

#map = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @scf_for_tensor_iter_args(%a: tensor<2048x512xf32>,
                                      %b: tensor<2048x512xf32>,
                                      %out: memref<2048x512xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    %init = tensor.empty() : tensor<2048x512xf32>

    %loop = scf.for %i = %c0 to %c2 step %c1
        iter_args(%acc = %init) -> (tensor<2048x512xf32>) {
      %tmp = tensor.empty() : tensor<2048x512xf32>

      %next = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]
      } ins(%acc, %a : tensor<2048x512xf32>, tensor<2048x512xf32>)
        outs(%tmp : tensor<2048x512xf32>) {
      ^bb0(%x: f32, %y: f32, %unused: f32):
        %sum = arith.addf %x, %y : f32
        linalg.yield %sum : f32
      } -> tensor<2048x512xf32>

      scf.yield %next : tensor<2048x512xf32>
    }

    %final_init = tensor.empty() : tensor<2048x512xf32>

    %result = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%loop, %b : tensor<2048x512xf32>, tensor<2048x512xf32>)
      outs(%final_init : tensor<2048x512xf32>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    } -> tensor<2048x512xf32>

    bufferization.materialize_in_destination
      %result in restrict writable %out
        : (tensor<2048x512xf32>, memref<2048x512xf32>) -> ()

    return
  }
}

// CHECK: shard.grid @nsp
// CHECK-SAME: shape = 16
// CHECK-LABEL: func.func @scf_for_tensor_iter_args

// The localized loop path slices external tensor values needed by the
// cloned loop and by the downstream localized consumer.
// CHECK-DAG: %[[INIT_TILE:.*]] = shard.all_slice
// CHECK-DAG: %[[A_TILE:.*]] = shard.all_slice
// CHECK-DAG: %[[B_TILE:.*]] = shard.all_slice

// The scf.for with tensor iter_args is cloned as a local tensor loop.
// The local loop carries tensor<128x512xf32>, not the original
// tensor<2048x512xf32>.
// CHECK: %[[LOCAL_LOOP:.*]] = scf.for
// CHECK-SAME: iter_args(%{{.*}} = %[[INIT_TILE]])
// CHECK-SAME: -> (tensor<128x512xf32>)
// CHECK-SAME: nsp.localized_loop

// The body of the localized loop contains the cloned local tensor computation.
// CHECK: tensor.empty() : tensor<128x512xf32>
// CHECK: linalg.generic
// CHECK: ins(%{{.*}}, %[[A_TILE]]
// CHECK: outs(%{{.*}} : tensor<128x512xf32>)
// CHECK: arith.addf
// CHECK: linalg.yield
// CHECK: scf.yield %{{.*}} : tensor<128x512xf32>

// The downstream consumer is localized and consumes the local loop result.
// CHECK: %[[FINAL_INIT:.*]] = tensor.empty() : tensor<128x512xf32>
// CHECK: %[[LOCAL_RESULT:.*]] = linalg.generic
// CHECK: ins(%[[LOCAL_LOOP]], %[[B_TILE]]
// CHECK: outs(%[[FINAL_INIT]]
// CHECK: -> tensor<128x512xf32>

// The final local tile is handed off to NSPMaterializePass.
// CHECK: nsp.materialize_tile %[[LOCAL_RESULT]] : tensor<128x512xf32> into %{{.*}} : memref<2048x512xf32>
// CHECK-SAME: grid @nsp
// CHECK-SAME: split_axis = 0
// CHECK-SAME: tile_shape = [128, 512]

// This is not the cyclic DOALL loop-distribution path.
// CHECK-NOT: nsp.distribution = "cyclic"
