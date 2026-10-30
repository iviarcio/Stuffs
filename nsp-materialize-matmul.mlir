// RUN: linalg-hexagon-opt %s -nsp-shard-planner -nsp-sharding-propagation -nsp-localize -nsp-materialize | FileCheck %s --check-prefix=MAT
// Verify destination subview offsets after tile materialization.
// The partitioned dimension must carry the dynamic offset; the other must be zero.
//
// Validate M-only and N-only matmul localization on an 8x4 physical NSP grid.
// The planner must use minTileExtent = 16. K is never partitioned.
// Each computation has a bufferization.materialize_in_destination sink,
// required by the matmul-specific localization path.
//
// The localized generic must carry the selected output split axis and
// the materialize_tile must use that same axis and local tile shape.
//
module {
  shard.grid @nsp(shape = 8x4)

  // CHECK-LABEL: func.func @m_larger_2048x1024(
  // CHECK: linalg.generic
  // CHECK: nsp.localized
  // CHECK: nsp.materialize_split_axis = 0
  // CHECK: nsp.materialize_tile_shape = array<i64: 64, 1024>
  // CHECK: nsp.materialize_tile
  // CHECK-SAME: split_axis = 0
  // CHECK-SAME: tile_shape = [64, 1024]
  // MAT-LABEL: func.func @m_larger_2048x1024(
  // MAT: memref.subview %dst[{{%[^,\]]+}}, 0] [64, 1024] [1, 1]
  // MAT-NOT: nsp.materialize_tile
  func.func @m_larger_2048x1024(%a: tensor<2048x512xf16>,
                         %b: tensor<512x1024xf16>,
                         %init: tensor<2048x1024xf16>, %dst: memref<2048x1024xf16>) {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(i, j, k) -> (i, k)>,
        affine_map<(i, j, k) -> (k, j)>,
        affine_map<(i, j, k) -> (i, j)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<2048x512xf16>, tensor<512x1024xf16>)
      outs(%init : tensor<2048x1024xf16>) {
      ^bb0(%lhs: f16, %rhs: f16, %acc: f16):
        %prod = arith.mulf %lhs, %rhs : f16
        %sum = arith.addf %acc, %prod : f16
        linalg.yield %sum : f16
    } -> tensor<2048x1024xf16>
    bufferization.materialize_in_destination %result in %dst restrict writable : (tensor<2048x1024xf16>, memref<2048x1024xf16>) -> ()
    return
  }

  // CHECK-LABEL: func.func @n_larger_1024x2048(
  // CHECK: linalg.generic
  // CHECK: nsp.localized
  // CHECK: nsp.materialize_split_axis = 1
  // CHECK: nsp.materialize_tile_shape = array<i64: 1024, 64>
  // CHECK: nsp.materialize_tile
  // CHECK-SAME: split_axis = 1
  // CHECK-SAME: tile_shape = [1024, 64]
  // MAT-LABEL: func.func @n_larger_1024x2048(
  // MAT: memref.subview %dst[0, {{%[^,\]]+}}] [1024, 64] [1, 1]
  // MAT-NOT: nsp.materialize_tile
  func.func @n_larger_1024x2048(%a: tensor<1024x512xf16>,
                         %b: tensor<512x2048xf16>,
                         %init: tensor<1024x2048xf16>, %dst: memref<1024x2048xf16>) {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(i, j, k) -> (i, k)>,
        affine_map<(i, j, k) -> (k, j)>,
        affine_map<(i, j, k) -> (i, j)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<1024x512xf16>, tensor<512x2048xf16>)
      outs(%init : tensor<1024x2048xf16>) {
      ^bb0(%lhs: f16, %rhs: f16, %acc: f16):
        %prod = arith.mulf %lhs, %rhs : f16
        %sum = arith.addf %acc, %prod : f16
        linalg.yield %sum : f16
    } -> tensor<1024x2048xf16>
    bufferization.materialize_in_destination %result in %dst restrict writable : (tensor<1024x2048xf16>, memref<1024x2048xf16>) -> ()
    return
  }

  // CHECK-LABEL: func.func @m_equal_n_1024x1024(
  // CHECK: linalg.generic
  // CHECK: nsp.localized
  // CHECK: nsp.materialize_split_axis = 0
  // CHECK: nsp.materialize_tile_shape = array<i64: 32, 1024>
  // CHECK: nsp.materialize_tile
  // CHECK-SAME: split_axis = 0
  // CHECK-SAME: tile_shape = [32, 1024]
  // MAT-LABEL: func.func @m_equal_n_1024x1024(
  // MAT: memref.subview %dst[{{%[^,\]]+}}, 0] [32, 1024] [1, 1]
  // MAT-NOT: nsp.materialize_tile
  func.func @m_equal_n_1024x1024(%a: tensor<1024x512xf16>,
                         %b: tensor<512x1024xf16>,
                         %init: tensor<1024x1024xf16>, %dst: memref<1024x1024xf16>) {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(i, j, k) -> (i, k)>,
        affine_map<(i, j, k) -> (k, j)>,
        affine_map<(i, j, k) -> (i, j)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<1024x512xf16>, tensor<512x1024xf16>)
      outs(%init : tensor<1024x1024xf16>) {
      ^bb0(%lhs: f16, %rhs: f16, %acc: f16):
        %prod = arith.mulf %lhs, %rhs : f16
        %sum = arith.addf %acc, %prod : f16
        linalg.yield %sum : f16
    } -> tensor<1024x1024xf16>
    bufferization.materialize_in_destination %result in %dst restrict writable : (tensor<1024x1024xf16>, memref<1024x1024xf16>) -> ()
    return
  }

  // CHECK-LABEL: func.func @m_small_32x2048(
  // CHECK: linalg.generic
  // CHECK: nsp.localized
  // CHECK: nsp.materialize_split_axis = 1
  // CHECK: nsp.materialize_tile_shape = array<i64: 32, 64>
  // CHECK: nsp.materialize_tile
  // CHECK-SAME: split_axis = 1
  // CHECK-SAME: tile_shape = [32, 64]
  // MAT-LABEL: func.func @m_small_32x2048(
  // MAT: memref.subview %dst[0, {{%[^,\]]+}}] [32, 64] [1, 1]
  // MAT-NOT: nsp.materialize_tile
  func.func @m_small_32x2048(%a: tensor<32x512xf16>,
                         %b: tensor<512x2048xf16>,
                         %init: tensor<32x2048xf16>, %dst: memref<32x2048xf16>) {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(i, j, k) -> (i, k)>,
        affine_map<(i, j, k) -> (k, j)>,
        affine_map<(i, j, k) -> (i, j)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<32x512xf16>, tensor<512x2048xf16>)
      outs(%init : tensor<32x2048xf16>) {
      ^bb0(%lhs: f16, %rhs: f16, %acc: f16):
        %prod = arith.mulf %lhs, %rhs : f16
        %sum = arith.addf %acc, %prod : f16
        linalg.yield %sum : f16
    } -> tensor<32x2048xf16>
    bufferization.materialize_in_destination %result in %dst restrict writable : (tensor<32x2048xf16>, memref<32x2048xf16>) -> ()
    return
  }

  // CHECK-LABEL: func.func @n_small_2048x32(
  // CHECK: linalg.generic
  // CHECK: nsp.localized
  // CHECK: nsp.materialize_split_axis = 0
  // CHECK: nsp.materialize_tile_shape = array<i64: 64, 32>
  // CHECK: nsp.materialize_tile
  // CHECK-SAME: split_axis = 0
  // CHECK-SAME: tile_shape = [64, 32]
  // MAT-LABEL: func.func @n_small_2048x32(
  // MAT: memref.subview %dst[{{%[^,\]]+}}, 0] [64, 32] [1, 1]
  // MAT-NOT: nsp.materialize_tile
  func.func @n_small_2048x32(%a: tensor<2048x512xf16>,
                         %b: tensor<512x32xf16>,
                         %init: tensor<2048x32xf16>, %dst: memref<2048x32xf16>) {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(i, j, k) -> (i, k)>,
        affine_map<(i, j, k) -> (k, j)>,
        affine_map<(i, j, k) -> (i, j)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<2048x512xf16>, tensor<512x32xf16>)
      outs(%init : tensor<2048x32xf16>) {
      ^bb0(%lhs: f16, %rhs: f16, %acc: f16):
        %prod = arith.mulf %lhs, %rhs : f16
        %sum = arith.addf %acc, %prod : f16
        linalg.yield %sum : f16
    } -> tensor<2048x32xf16>
    bufferization.materialize_in_destination %result in %dst restrict writable : (tensor<2048x32xf16>, memref<2048x32xf16>) -> ()
    return
  }

  // CHECK-LABEL: func.func @min_extent_512x256(
  // CHECK: linalg.generic
  // CHECK: nsp.localized
  // CHECK: nsp.materialize_split_axis = 0
  // CHECK: nsp.materialize_tile_shape = array<i64: 16, 256>
  // CHECK: nsp.materialize_tile
  // CHECK-SAME: split_axis = 0
  // CHECK-SAME: tile_shape = [16, 256]
  // MAT-LABEL: func.func @min_extent_512x256(
  // MAT: memref.subview %dst[{{%[^,\]]+}}, 0] [16, 256] [1, 1]
  // MAT-NOT: nsp.materialize_tile
  func.func @min_extent_512x256(%a: tensor<512x256xf16>,
                         %b: tensor<256x256xf16>,
                         %init: tensor<512x256xf16>, %dst: memref<512x256xf16>) {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(i, j, k) -> (i, k)>,
        affine_map<(i, j, k) -> (k, j)>,
        affine_map<(i, j, k) -> (i, j)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<512x256xf16>, tensor<256x256xf16>)
      outs(%init : tensor<512x256xf16>) {
      ^bb0(%lhs: f16, %rhs: f16, %acc: f16):
        %prod = arith.mulf %lhs, %rhs : f16
        %sum = arith.addf %acc, %prod : f16
        linalg.yield %sum : f16
    } -> tensor<512x256xf16>
    bufferization.materialize_in_destination %result in %dst restrict writable : (tensor<512x256xf16>, memref<512x256xf16>) -> ()
    return
  }

  // CHECK-LABEL: func.func @below_min_extent_256x256(
  // CHECK-NOT: nsp.localized
  // CHECK: linalg.generic
  // CHECK-NOT: nsp.materialize_tile
  // MAT-LABEL: func.func @below_min_extent_256x256(
  // MAT-NOT: nsp.materialize_tile
  // MAT: linalg.generic
  func.func @below_min_extent_256x256(%a: tensor<256x256xf16>,
                         %b: tensor<256x256xf16>,
                         %init: tensor<256x256xf16>, %dst: memref<256x256xf16>) {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(i, j, k) -> (i, k)>,
        affine_map<(i, j, k) -> (k, j)>,
        affine_map<(i, j, k) -> (i, j)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<256x256xf16>, tensor<256x256xf16>)
      outs(%init : tensor<256x256xf16>) {
      ^bb0(%lhs: f16, %rhs: f16, %acc: f16):
        %prod = arith.mulf %lhs, %rhs : f16
        %sum = arith.addf %acc, %prod : f16
        linalg.yield %sum : f16
    } -> tensor<256x256xf16>
    bufferization.materialize_in_destination %result in %dst restrict writable : (tensor<256x256xf16>, memref<256x256xf16>) -> ()
    return
  }
}
