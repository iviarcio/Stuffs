// RUN: linalg-hexagon-opt %s -nsp-shard-planner | FileCheck %s
//
// Verify matmul-like linalg.generic planning using a 2-D physical NSP grid
// with 32 participants (8 cores x 4 threads) and minTileExtent = 16.
// Only one output dimension is sharded; the K reduction stays local.
//
// M-only: A[[0,1],[]], B[[],[]], C[[0,1],[]].
// N-only: A[[],[]], B[[],[0,1]], C[[],[0,1]].
//
module {
  shard.grid @nsp(shape = 8x4)

  // CHECK-LABEL: func.func @m_larger_2048x1024(
  // CHECK-DAG: shard.sharding @nsp split_axes = [[0, 1], []]
  // CHECK-DAG: shard.sharding @nsp split_axes = [[], []]
  // CHECK: linalg.generic
  func.func @m_larger_2048x1024(%a: tensor<2048x512xf32>,
                         %b: tensor<512x1024xf32>,
                         %init: tensor<2048x1024xf32>) -> tensor<2048x1024xf32> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(i, j, k) -> (i, k)>,
        affine_map<(i, j, k) -> (k, j)>,
        affine_map<(i, j, k) -> (i, j)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<2048x512xf32>, tensor<512x1024xf32>)
      outs(%init : tensor<2048x1024xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
        %prod = arith.mulf %lhs, %rhs : f32
        %sum = arith.addf %acc, %prod : f32
        linalg.yield %sum : f32
    } -> tensor<2048x1024xf32>
    return %result : tensor<2048x1024xf32>
  }

  // CHECK-LABEL: func.func @n_larger_1024x2048(
  // CHECK-DAG: shard.sharding @nsp split_axes = [[], [0, 1]]
  // CHECK-DAG: shard.sharding @nsp split_axes = [[], []]
  // CHECK: linalg.generic
  func.func @n_larger_1024x2048(%a: tensor<1024x512xf32>,
                         %b: tensor<512x2048xf32>,
                         %init: tensor<1024x2048xf32>) -> tensor<1024x2048xf32> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(i, j, k) -> (i, k)>,
        affine_map<(i, j, k) -> (k, j)>,
        affine_map<(i, j, k) -> (i, j)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<1024x512xf32>, tensor<512x2048xf32>)
      outs(%init : tensor<1024x2048xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
        %prod = arith.mulf %lhs, %rhs : f32
        %sum = arith.addf %acc, %prod : f32
        linalg.yield %sum : f32
    } -> tensor<1024x2048xf32>
    return %result : tensor<1024x2048xf32>
  }

  // CHECK-LABEL: func.func @m_equal_n_1024x1024(
  // CHECK-DAG: shard.sharding @nsp split_axes = [[0, 1], []]
  // CHECK-DAG: shard.sharding @nsp split_axes = [[], []]
  // CHECK: linalg.generic
  func.func @m_equal_n_1024x1024(%a: tensor<1024x512xf32>,
                         %b: tensor<512x1024xf32>,
                         %init: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(i, j, k) -> (i, k)>,
        affine_map<(i, j, k) -> (k, j)>,
        affine_map<(i, j, k) -> (i, j)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<1024x512xf32>, tensor<512x1024xf32>)
      outs(%init : tensor<1024x1024xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
        %prod = arith.mulf %lhs, %rhs : f32
        %sum = arith.addf %acc, %prod : f32
        linalg.yield %sum : f32
    } -> tensor<1024x1024xf32>
    return %result : tensor<1024x1024xf32>
  }

  // CHECK-LABEL: func.func @m_small_32x2048(
  // CHECK-DAG: shard.sharding @nsp split_axes = [[], [0, 1]]
  // CHECK-DAG: shard.sharding @nsp split_axes = [[], []]
  // CHECK: linalg.generic
  func.func @m_small_32x2048(%a: tensor<32x512xf32>,
                         %b: tensor<512x2048xf32>,
                         %init: tensor<32x2048xf32>) -> tensor<32x2048xf32> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(i, j, k) -> (i, k)>,
        affine_map<(i, j, k) -> (k, j)>,
        affine_map<(i, j, k) -> (i, j)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<32x512xf32>, tensor<512x2048xf32>)
      outs(%init : tensor<32x2048xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
        %prod = arith.mulf %lhs, %rhs : f32
        %sum = arith.addf %acc, %prod : f32
        linalg.yield %sum : f32
    } -> tensor<32x2048xf32>
    return %result : tensor<32x2048xf32>
  }

  // CHECK-LABEL: func.func @n_small_2048x32(
  // CHECK-DAG: shard.sharding @nsp split_axes = [[0, 1], []]
  // CHECK-DAG: shard.sharding @nsp split_axes = [[], []]
  // CHECK: linalg.generic
  func.func @n_small_2048x32(%a: tensor<2048x512xf32>,
                         %b: tensor<512x32xf32>,
                         %init: tensor<2048x32xf32>) -> tensor<2048x32xf32> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(i, j, k) -> (i, k)>,
        affine_map<(i, j, k) -> (k, j)>,
        affine_map<(i, j, k) -> (i, j)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<2048x512xf32>, tensor<512x32xf32>)
      outs(%init : tensor<2048x32xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
        %prod = arith.mulf %lhs, %rhs : f32
        %sum = arith.addf %acc, %prod : f32
        linalg.yield %sum : f32
    } -> tensor<2048x32xf32>
    return %result : tensor<2048x32xf32>
  }

  // CHECK-LABEL: func.func @m_larger_but_not_divisible_1000x2048(
  // CHECK-DAG: shard.sharding @nsp split_axes = [[], [0, 1]]
  // CHECK-DAG: shard.sharding @nsp split_axes = [[], []]
  // CHECK: linalg.generic
  func.func @m_larger_but_not_divisible_1000x2048(%a: tensor<1000x512xf32>,
                         %b: tensor<512x2048xf32>,
                         %init: tensor<1000x2048xf32>) -> tensor<1000x2048xf32> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(i, j, k) -> (i, k)>,
        affine_map<(i, j, k) -> (k, j)>,
        affine_map<(i, j, k) -> (i, j)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<1000x512xf32>, tensor<512x2048xf32>)
      outs(%init : tensor<1000x2048xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
        %prod = arith.mulf %lhs, %rhs : f32
        %sum = arith.addf %acc, %prod : f32
        linalg.yield %sum : f32
    } -> tensor<1000x2048xf32>
    return %result : tensor<1000x2048xf32>
  }

  // CHECK-LABEL: func.func @min_extent_512x256(
  // CHECK-DAG: shard.sharding @nsp split_axes = [[0, 1], []]
  // CHECK-DAG: shard.sharding @nsp split_axes = [[], []]
  // CHECK: linalg.generic
  func.func @min_extent_512x256(%a: tensor<512x256xf32>,
                         %b: tensor<256x256xf32>,
                         %init: tensor<512x256xf32>) -> tensor<512x256xf32> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(i, j, k) -> (i, k)>,
        affine_map<(i, j, k) -> (k, j)>,
        affine_map<(i, j, k) -> (i, j)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<512x256xf32>, tensor<256x256xf32>)
      outs(%init : tensor<512x256xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
        %prod = arith.mulf %lhs, %rhs : f32
        %sum = arith.addf %acc, %prod : f32
        linalg.yield %sum : f32
    } -> tensor<512x256xf32>
    return %result : tensor<512x256xf32>
  }

  // CHECK-LABEL: func.func @below_min_extent_256x256(
  // CHECK-NOT: shard.sharding
  // CHECK: linalg.generic
  // CHECK-NOT: shard.shard
  func.func @below_min_extent_256x256(%a: tensor<256x256xf32>,
                         %b: tensor<256x256xf32>,
                         %init: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(i, j, k) -> (i, k)>,
        affine_map<(i, j, k) -> (k, j)>,
        affine_map<(i, j, k) -> (i, j)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<256x256xf32>, tensor<256x256xf32>)
      outs(%init : tensor<256x256xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
        %prod = arith.mulf %lhs, %rhs : f32
        %sum = arith.addf %acc, %prod : f32
        linalg.yield %sum : f32
    } -> tensor<256x256xf32>
    return %result : tensor<256x256xf32>
  }

  // CHECK-LABEL: func.func @tiny_32x32(
  // CHECK-NOT: shard.sharding
  // CHECK: linalg.generic
  // CHECK-NOT: shard.shard
  func.func @tiny_32x32(%a: tensor<32x64xf32>,
                         %b: tensor<64x32xf32>,
                         %init: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(i, j, k) -> (i, k)>,
        affine_map<(i, j, k) -> (k, j)>,
        affine_map<(i, j, k) -> (i, j)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<32x64xf32>, tensor<64x32xf32>)
      outs(%init : tensor<32x32xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
        %prod = arith.mulf %lhs, %rhs : f32
        %sum = arith.addf %acc, %prod : f32
        linalg.yield %sum : f32
    } -> tensor<32x32xf32>
    return %result : tensor<32x32xf32>
  }

  // CHECK-LABEL: func.func @preferred_not_divisible_1056x1024(
  // CHECK-DAG: shard.sharding @nsp split_axes = [[], [0, 1]]
  // CHECK-DAG: shard.sharding @nsp split_axes = [[], []]
  // CHECK: linalg.generic
  func.func @preferred_not_divisible_1056x1024(%a: tensor<1056x256xf32>,
                         %b: tensor<256x1024xf32>,
                         %init: tensor<1056x1024xf32>) -> tensor<1056x1024xf32> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(i, j, k) -> (i, k)>,
        affine_map<(i, j, k) -> (k, j)>,
        affine_map<(i, j, k) -> (i, j)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<1056x256xf32>, tensor<256x1024xf32>)
      outs(%init : tensor<1056x1024xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
        %prod = arith.mulf %lhs, %rhs : f32
        %sum = arith.addf %acc, %prod : f32
        linalg.yield %sum : f32
    } -> tensor<1056x1024xf32>
    return %result : tensor<1056x1024xf32>
  }

}
