// RUN: linalg-hexagon-opt %s -nsp-localize | FileCheck %s

// This test validates the DOALL loop-distribution path in NSPLocalizePass.
// Input:
//   A plain scf.for loop with a per-iteration memref.store whose destination
//   index depends directly on the induction variable.
//
// Expected behavior in non-collective mode:
//   - recognize the loop as a DOALL-style output loop;
//   - create shard.process_linear_index on @nsp;
//   - rewrite the lower bound as:
//       newLb = lb + processLinearIndex * step
//   - rewrite the step as:
//       newStep = step * numShards
//   - clone the original loop body into the distributed loop;
//   - mark the new loop with:
//       nsp.distribution = "cyclic"

module {
  shard.grid @nsp(shape = 16)

  func.func @doall_memref_store(%input: memref<1024xf32>,
                                %output: memref<1024xf32>) {
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %one = arith.constant 1.000000e+00 : f32

    scf.for %i = %c0 to %c1024 step %c1 {
      %x = memref.load %input[%i] : memref<1024xf32>
      %y = arith.addf %x, %one : f32
      memref.store %y, %output[%i] : memref<1024xf32>
    }

    return
  }
}

// CHECK-LABEL: func.func @doall_memref_store(
// CHECK-SAME: %[[INPUT:.*]]: memref<1024xf32>
// CHECK-SAME: %[[OUTPUT:.*]]: memref<1024xf32>

// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[UB:.*]] = arith.constant 1024 : index
// CHECK-DAG: %[[STEP:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32

// The distributed loop lower bound is:
//   lb + shard.process_linear_index * step
// CHECK: %[[PID:.*]] = shard.process_linear_index on @nsp : index
// CHECK: %[[OFFSET:.*]] = arith.muli %[[PID]], %[[STEP]] : index
// CHECK: %[[NEW_LB:.*]] = arith.addi %[[C0]], %[[OFFSET]] : index

// The distributed loop step is:
//   step * 16
// CHECK: %[[NUM_SHARDS:.*]] = arith.constant 16 : index
// CHECK: %[[NEW_STEP:.*]] = arith.muli %[[STEP]], %[[NUM_SHARDS]] : index

// The original loop is replaced by a cyclically distributed loop.
// CHECK: scf.for %[[IV:.*]] = %[[NEW_LB]] to %[[UB]] step %[[NEW_STEP]]
// CHECK: nsp.distribution = "cyclic"

// The loop body is preserved and remapped to the new induction variable.
// CHECK: %[[X:.*]] = memref.load %[[INPUT]][%[[IV]]] : memref<1024xf32>
// CHECK: %[[Y:.*]] = arith.addf %[[X]], %[[ONE]] : f32
// CHECK: memref.store %[[Y]], %[[OUTPUT]][%[[IV]]] : memref<1024xf32>

// No tensor-localization hand-off should be produced by this test.
// CHECK-NOT: nsp.materialize_tile
// CHECK-NOT: shard.all_slice
