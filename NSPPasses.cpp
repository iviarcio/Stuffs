//===- NSPPasses.cpp - Sharding and SPMDization Passes in MLIR ------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Registration and construction o f the NSP "nsp-shard" pass pipeline.
//
// This file is responsible for:
//   1. Declaring forward references to pass factories defined in sibling files
//      (NSPShardPlanner.cpp, NSPShardingPropagationPass.cpp,
//       NSPLocalizePass.cpp, NSPMaterializePass.cpp).
//   2. Defining the composite pipeline that sequences these passes with
//      canonicalization/CSE boundaries.
//   3. Registering individual passes and the pipeline with the MLIR registries
//      so they are accessible from tools (linalg-hexagon-opt, mlir-opt).
//   4. Registering NSP dialecte xtensions (ShardingInterface external models)
//      that enable propagation to reason about non-shard ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

// Useful common transforms.
#include "mlir/Transforms/Passes.h" // canonicalizer, cse

// shard dialect IR (GridOp, ShardOp, collectives, attrs).
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shard/IR/ShardDialect.h"
#include "mlir/Dialect/Shard/IR/ShardOps.h"
#include "mlir/Dialect/Shard/Transforms/Passes.h"

#include "hexagon/NSP/NSPPasses.h"

// NSP pieces.
namespace mlir {
namespace hexagon {

std::unique_ptr<Pass> createNSPShardPlannerPass();
std::unique_ptr<Pass> createNSPShardPlannerPass(int64_t nspCount,
                                                bool shardingAllowCollectives);
std::unique_ptr<Pass> createNSPShardPlannerPass(int64_t numCores,
                                                int64_t numThreads,
                                                bool shardingAllowCollectives);

std::unique_ptr<Pass> createNSPShardingPropagationPass();

std::unique_ptr<Pass> createNSPLocalizePass();
std::unique_ptr<Pass> createNSPLocalizePass(bool shardingAllowCollectives);

std::unique_ptr<Pass> createNSPMaterializePass();

// From NSPShardInterfaceModels.cpp (attach external models).
void registerNSPShardInterfaceModels(DialectRegistry &registry);

} // namespace hexagon
} // namespace mlir

using namespace mlir;
using namespace hexagon;

namespace {

//===----------------------------------------------------------------------===//
// Pipeline options (exposed as command-line flags).
//
// Usage:
//   --pass-pipeline="nsp-shard(nsp-count=32 propagate=true allow-collectives)"
//   --pass-pipeline="nsp-shard(nsp-cores=8 nsp-threads=4)"
//===----------------------------------------------------------------------===//
struct NSPShardPipelineOptions
    : public PassPipelineOptions<NSPShardPipelineOptions> {
  Option<int64_t> nspCount{
      *this, "nsp-count",
      llvm::cl::desc(
          "Legacy flattened number of NSP participants in the mesh"),
      llvm::cl::init(32)};

  Option<int64_t> nspCores{
      *this, "nsp-cores",
      llvm::cl::desc(
          "Number of NSP cores in grid axis 0. If unset, nsp-count is used."),
      llvm::cl::init(0)};

  Option<int64_t> nspThreads{
      *this, "nsp-threads",
      llvm::cl::desc(
          "Number of NSP threads per core in grid axis 1. If unset, uses 1."),
      llvm::cl::init(0)};

  Option<bool> runPropagation{
      *this, "propagate",
      llvm::cl::desc("Run sharding propagation after planning"),
      llvm::cl::init(true)};

  Option<bool> allowCollectives{
      *this, "allow-collectives",
      llvm::cl::desc(
          "Allow planner/SPMDization to insert collectives (e.g. all-reduce)"),
      llvm::cl::init(false)};

  Option<bool> canonicalize{
      *this, "canonicalize",
      llvm::cl::desc("Run canonicalize + CSE at key boundaries"),
      llvm::cl::init(true)};
};

//===--------------------------------------------------------------------===//
// Pipeline construction.
//
// The pipeline transforms linalg-on-tensor IR into partitioned SPMD code:
//
//   Input:   linalg.generic (tensor semantics, single-core)
//   Output:  per-participant local compute + memref.subview destination writes
//
// The split between localization (step 3) and materialization (step 4) is
// intentional: downstream tiling/vectorization runs between them while the IR
// still has pure tensor semantics.
//
//  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌───────────────┐
//  │ 1. Planner   │-->│ 2. Propagate │-->│ 3. Localize  │-->│ 4.Materialize │
//  └──────────────┘   └──────────────┘   └──────────────┘   └───────────────┘
//         │                 │                  │                  │
//  shard.grid/shard   infer missing      local tensor +     memref.subview
//  annotations        shardings          nsp.materialize_   writes
//                                        tile
//===--------------------------------------------------------------------===//
static void buildNSPShardPipeline(OpPassManager &pm,
                                  const NSPShardPipelineOptions &opts) {

  // Step 1: Planner - attach shard annotations to linalg.generic ops.
  // Runs on func.func (analyzes per-function).
  //
  // Preserve the legacy nsp-count behavior by default, but allow callers to
  // expose the hardware topology explicitly as a 2-D grid: [cores, threads].
  const int64_t numCores = opts.nspCores > 0 ? opts.nspCores : opts.nspCount;
  const int64_t numThreads = opts.nspThreads > 0 ? opts.nspThreads : 1;

  pm.addNestedPass<func::FuncOp>(createNSPShardPlannerPass(
      numCores, numThreads, opts.allowCollectives));
  if (opts.canonicalize) {
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }

  // Step 2: Propagation - infer shardings for unannotated operands by
  // traversing the dataflow graph (backward then forward). Relies on
  // ShardingInterface external models registered by
  // registerNSPShardInterfaceModels().
  if (opts.runPropagation) {
    pm.addNestedPass<func::FuncOp>(createNSPShardingPropagationPass());
  }
  if (opts.canonicalize) {
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }

  // Step 3: Localization - rewrite sharded compute into per-participant
  // local tensor ops. Emits nsp.materialize_tile as a hand-off to step 4.
  // In non-collective mode, also distributes scf.for loops cyclically.
  //
  // After this point the IR still has tensor semantics, so tiling and
  // vectorization passes can be inserted here before materialization.
  pm.addPass(createNSPLocalizePass(opts.allowCollectives));

  // Optional cleanup boundary before downstream tiling/vectorization
  if (opts.canonicalize) {
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }

  // Step 4: Materialization - lower nsp.materialize_tile into explicit
  // per-participant memref.subview stores. After this pass, the NSP
  // dialect ops are fully consumed.
  pm.addPass(createNSPMaterializePass());

  // 5. Cleanup
  if (opts.canonicalize) {
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// Public registration entrypoints.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace hexagon {

/// Register individual NSP passes for standalone use (e.g. in lit tests):
///    -nsp-shard-planner
///    -nsp-sharding-propagation
///    -nsp-localize
///    -nsp-materialize
void registerNSPPasses() {
  // The lambda form avoids needing the pass type visible here.
  registerPass(
      []() -> std::unique_ptr<Pass> { return createNSPShardPlannerPass(); });
  registerPass([]() -> std::unique_ptr<Pass> {
    return createNSPShardingPropagationPass();
  });
  registerPass(
      []() -> std::unique_ptr<Pass> { return createNSPLocalizePass(); });
  registerPass(
      []() -> std::unique_ptr<Pass> { return createNSPMaterializePass(); });
}

/// Register NSP pipelines.
void registerNSPPipelines() {
  PassPipelineRegistration<NSPShardPipelineOptions>(
      "nsp-shard",
      "NSP pipeline: shard planning -> Propagation -> Localization -> "
      "Materialization",
      buildNSPShardPipeline);
}

/// Register dialect extensions required by NSP sharding propagation.
void registerNSPDialectExtensions(DialectRegistry &registry) {
  registry.insert<mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::shard::ShardDialect, mlir::tensor::TensorDialect,
                  mlir::LLVM::LLVMDialect>();

  registerNSPShardInterfaceModels(registry);
}

} // namespace hexagon
} // namespace mlir
