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
                                                bool allowCollectives);

std::unique_ptr<Pass> createNSPShardingPropagationPass();

std::unique_ptr<Pass> createNSPLocalizePass();
std::unique_ptr<Pass> createNSPLocalizePass(bool allowCollectives);

std::unique_ptr<Pass> createNSPMaterializePass();

// From NSPShardInterfaceModels.cpp (attach external models).
void registerNSPShardInterfaceModels(DialectRegistry &registry);

} // namespace hexagon
} // namespace mlir

using namespace mlir;
using namespace hexagon;

namespace {

// Pipeline options:
//   -nsp-shard="nsp-count=16 allow-collectives"
struct NSPShardPipelineOptions
    : public PassPipelineOptions<NSPShardPipelineOptions> {
  Option<int64_t> nspCount{*this, "nsp-count",
                           llvm::cl::desc("Number of NSP devices in the mesh"),
                           llvm::cl::init(16)};

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

/// Build the canonical NSP shard pipeline. At high-level:
///  1. Planner: attach shard annotations (device mesh + tensor layout).
///  2. Propagation: infer missing shardings using ShardingInterface.
///  3. Localization: rewrite supported shard-annotated compute into
///     per-NSP local tensor compute while preserving pure tensor semantics.
///  4. Materialization: connect localized results to destination buffers
///     using explicit per-NSP subviews / store-by-tile rewrites.
///  5. Cleanup: canonicalize/CSE.
///
/// Note:
///   The split between localization and materialization is intentional:
///   tiling/vectorization may run in between these two passes, while the IR
///   still has pure tensor semantics after localization.
static void buildNSPShardPipeline(OpPassManager &pm,
                                  const NSPShardPipelineOptions &opts) {

  // 1. Planner (NSPShardPlannerPass in NSPShardPlanner.cpp. Runs on func.func)
  pm.addNestedPass<func::FuncOp>(
      createNSPShardPlannerPass(opts.nspCount, opts.allowCollectives));
  if (opts.canonicalize) {
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }

  // 2. NSPSharding propagation (forked from Shard dialect. Runs on func.func)
  // This pass relies on shard::ShardingInterface models for ops in the graph.
  // NSPShardInterfaceModels.cpp attaches missing models.
  if (opts.runPropagation) {
    pm.addNestedPass<func::FuncOp>(createNSPShardingPropagationPass());
  }
  if (opts.canonicalize) {
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }

  // 3. Localization
  // - rewrite supported shard-annotated compute into per-NSP local tensor
  //   compute
  // - preserve pure tensor semantics so downstream tiling/vectorization can
  //   still match linalg ops.
  pm.addPass(createNSPLocalizePass(opts.allowCollectives));

  // Optional cleanup boundary before downstream tiling/vectorization
  if (opts.canonicalize) {
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }

  // NOTE:
  // Hexagon tiling/vectorization may be inserted here in a real pipeline
  // stage, between NSP localization and NSP Materialization.

  // 4. Materialization
  // - reconnect localized results to destination buffers,
  // - compute per-NSP offsets via shard.process_linear_index,
  // - rewrite sinks into explicit subviews / store-by-tile forms.
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

/// Register NSP passes (so tools can do: -nsp-shard-planner -nsp-spmdize ...).
///
/// In the future we'll using TableGen, i.e., auto-generate these via
/// Passes.td and call registerNSPPasses() from that generated file.
/// For now, we use an explicit style that works well for tests.
void registerNSPPasses() {
  // The lambda form avoids needing the pass type visible here.
  registerPass(
      []() -> std::unique_ptr<Pass> { return createNSPShardPlannerPass(); });
  registerPass(
      []() -> std::unique_ptr<Pass> {
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
      "NSP pipeline: shard planning -> Propagation -> Localization -> Materialization",
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
