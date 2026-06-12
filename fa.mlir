#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module attributes {llvm.target_triple = "hexagon"} {
  shard.grid @nsp(shape = 32)
  func.func @attention_fwd_kernel(%argo: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: memref<*xf32> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %cst = arith.constant 0xFF800000 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c16_i32 = arith.constant 16 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %cst_1 = arith.constant -1.000000e+14 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c64 = arith.constant 64 : index
    %c0_i32 = arith.constant 0 : i32
    %c16_i64 = arith.constant 16 : i64
    %cst_3 = arith.constant 0.72134751 : f32
    %0 = tensor.empty() : tensor<32x16xf32>
    %1 = tensor.empty() : tensor<32xf32>
    %2 = linalg.fill ins(%cst_2 : f32) outs (%1 : tensor<32xf32>) -> tensor<32xf32>
    %3 = linalg.fill ins(%cst_1 : f32) outs (%1 : tensor<32xf32>) -> tensor<32xf32>
    %4 = tensor.empty() : tensor<32x64xf32>
    %5 = linalg.fill ins(%cst_0 : f32) outs (%4 : tensor<32x64xf32>) -> tensor<32x64xf32>
    %reinterpret_cast = memref.reinterpret_cast %argo to offset: [0], sizes: [32, 64], strides: [64, 1] : memref<*xf32> to memref<32x64xf32, strided<[64, 1]>>
    %6 = bufferization.to_tensor %reinterpret_cast restrict: memref<32x64xf32, strided<[64, 1]>> to tensor<32x64xf32>
    %all_slice = shard.all_slice %5 on @nsp grid_axes = [0] slice_axis = 0 : tensor<32x64xf32> -> tensor<1x64xf32>
    %all_slice_4 = shard.all_slice %2 on @nsp grid_axes = [0] slice_axis = 0 : tensor<32xf32> -> tensor<1xf32>
    %all_slice_5 = shard.all_slice %3 on @nsp grid_axes = [0] slice_axis = 0 : tensor<32xf32> -> tensor<1xf32>
    %all_slice_6 = shard.all_slice %0 on @nsp grid_axes = [0] slice_axis = 0 : tensor<32x16xf32> -> tensor<1x16xf32>
    %all_slice_7 = shard.all_slice %5 on @nsp grid_axes = [0] slice_axis = 0 : tensor<32x64xf32> -> tensor<1x64xf32>
    %all_slice_8 = shard.all_slice %2 on @nsp grid_axes = [0] slice_axis = 0 : tensor<32xf32> -> tensor<1xf32>
    %all_slice_9 = shard.all_slice %3 on @nsp grid_axes = [0] slice_axis = 0 : tensor<32xf32> -> tensor<1xf32>
    %all_slice_10 = shard.all_slice %0 on @nsp grid_axes = [0] slice_axis = 0 : tensor<32x16xf32> -> tensor<1x16xf32>    
    %7:5 = scf.for %arg10 = %c0_i32 to %c1024_i32 step %c16_i32 iter_args(%arg11 = %5, %arg12 = %2, %arg13 = %3, %arg14 = %c0_i64, %arg15 = %c0_i64) -> (tensor<32x64xf32>, tensor<32xf32>, tensor<32xf32>, i64, i64) {
      %10 = arith.index_cast %arg14 : i64 to index
      %11 = arith.muli %10, %c64 : index
      %reinterpret_cast_14 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [16, 64], strides: [64, 1] : memref<*xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
      %12 = bufferization.to_tensor %reinterpret_cast_14 restrict: memref<16x64xf32, strided<[64, 1], offset: ?>> to tensor<16x64xf32>
      %13 = tensor.empty() : tensor<64x16xf32>
      %transposed = linalg.transpose ins(%12 : tensor<16x64xf32>) outs (%13 : tensor<64x16xf32>) permutation = [1, 0]
      %14 = linalg.fill ins(%cst_0 : f32) outs (%0 : tensor<32x16xf32>) -> tensor<32x16xf32>
      %15 = linalg.matmul ins(%6, %transposed : tensor<32x64xf32>, tensor<64x16xf32>) outs(%14 : tensor<32x16xf32>) -> tensor<32x16xf32>
      %16 = linalg.fill ins(%cst : f32) outs(%1 : tensor<32xf32>) -> tensor<32xf32>
      %reduced = linalg.reduce ins(%15 : tensor<32x16xf32>) outs(%16 : tensor<32xf32>) dimensions = [1] (%in: f32, %init: f32) {
        %28 = arith.maxnumf %in, %init : f32
        linalg.yield %28 : f32
      }
      %17 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg13, %reduced : tensor<32xf32>, tensor<32xf32>) outs(%1 : tensor<32xf32>) {
        ^bb0(%in: f32, %in_17: f32, %out: f32):
          %28 = arith.mulf %in_17, %cst_3
          %29 = arith.maxnumf %in, %28 : f32
          linalg.yield %29 : f32
      } -> tensor<32xf32>
      %18 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%15, %17 : tensor<32x16xf32>, tensor<32xf32>) outs (%0 : tensor<32x16xf32>) {
        ^bb0(%in: f32, %in_17: f32, %out: f32):
          %28 = arith.mulf %in, %cst_3
          %29 = arith.subf %28, %in_17
          %30 = math.exp2 %29
          linalg.yield %30 : f32
      } -> tensor<32x16xf32>
      %19 = linalg.fill ins(%cst_0 : f32) outs (%1 : tensor<32xf32>) -> tensor<32xf32>
      %reduced_15 = linalg.reduce ins(%18 : tensor<32x16xf32>) outs(%19 : tensor<32xf32>) dimensions = [1] (%in: f32, %init: f32) {
        %28 = arith.addf %in, %init : f32
        linalg.yield %28 : f32
      }
      %20:2 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%arg12, %arg13, %17, %reduced_15 : tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) outs (%1, %1 : tensor<32xf32>, tensor<32xf32>) {
        ^bb0(%in: f32, %in_17: f32, %in_18: f32, %in_19: f32, %out: f32, %out_20: f32):
          %28 = arith.subf %in_17, %in_18
          %29 = math.exp2 %28
          %30 = arith.mulf %in, %29
          %31 = arith.addf %30, %in_19
          linalg.yield %29, %31 : f32, f32
      } -> (tensor<32xf32>, tensor<32xf32>)
      %21 = arith.index_cast %arg15 : i64 to index
      %22 = arith.muli %21, %c64 : index
      %reinterpret_cast_16 = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [16, 64], strides: [64, 1] : memref<*xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
      %23 = bufferization.to_tensor %reinterpret_cast_16 restrict: memref<16x64xf32, strided<[64, 1], offset: ?>> to tensor<16x64xf32>
      %24 = linalg.matmul ins(%18, %23 : tensor<32x16xf32>, tensor<16x64xf32>) outs(%5 : tensor<32x64xf32>) -> tensor<32x64xf32>
      %25 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20#0, %24 : tensor<32x64xf32>, tensor<32xf32>, tensor<32x64xf32>) outs (%4 : tensor<32x64xf32>) {
        ^bb0(%in: f32, %in_17: f32, %in_18: f32, %out: f32):
          %28 = arith.mulf %in, %in_17
          %29 = arith.addf %28, %in_18
          linalg.yield %29 : f32
      } -> tensor<32x64xf32>
      %26 = arith.addi %arg15, %c16_i64 : i64
      %27 = arith.addi %arg14, %c16_i64 : i64
      scf.yield %25, %20#1, %17, %27, %26 : tensor<32x64xf32>, tensor<32xf32>, tensor<32xf32>, i64, i64
    }
    %all_slice_11 = shard.all_slice %7#0 on @nsp grid_axes = [0] slice_axis = 0 : tensor<32x64xf32> -> tensor<1x64xf32>
    %all_slice_12 = shard.all_slice %7#1 on @nsp grid_axes = [0] slice_axis = 0 : tensor<32xf32> -> tensor<1xf32>
    %8 = tensor.empty() : tensor<1x64xf32>
    %9 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%all_slice_11, %all_slice_12 : tensor<1x64xf32>, tensor<1xf32>) outs (%8 : tensor<1x64xf32>) {
      ^bb0(%in: f32, %in_14: f32, %out: f32):
        %10 = arith.divf %cst_2, %in_14
        %11 = arith.mulf %in, %10
        linalg.yield %11 : f32
    } -> tensor<1x64xf32>
    %reinterpret_cast_13 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [32, 64], strides: [64, 1] : memref<*xf32> to memref<32x64xf32, strided<[64, 1]>>
    nsp.materialize_tile %9 : tensor<1x64xf32> into %reinterpret_cast_13 : memref<32x64xf32, strided<[64, 1]>> grid @nsp split_axis = 0 tile_shape = [1, 64]
    return
  }
}
