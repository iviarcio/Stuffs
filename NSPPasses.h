//===- NSPPasses.h --------------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Public entry points for registering NSP passes, pipelines, and dialect
// extensions required for NSP sharding and planning.
//
// This includes support for:
//  - Sharding (planner creation)
//  - Propagation
//  - Localization
//  - Materialization
//  - NSP-specific dialect extensions
//
// This header is intended for use by tools and drivers (e.g., *-opt),
// as well as by the Hexagon backend initialization code responsible for
// setting up the DialectRegistry and registering pass pipelines.
//
//===----------------------------------------------------------------------===//

#ifndef QCOM_HEXAGON_BACKEND_NSP_NSPPASSES_H
#define QCOM_HEXAGON_BACKEND_NSP_NSPPASSES_H

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace hexagon {

//===----------------------------------------------------------------------===//
// Pass factories
//===----------------------------------------------------------------------===//

/// Create the NSP shard planner pass with default policy (16 NSPs, no
/// collectives). Analyzes linalg generic ops within a func.func and attaches
/// shard dialect annotations describing how each tensor operand is partitioned.
std::unique_ptr<mlir::Pass> createNSPShardPlannerPass();

/// Create the planner pass with an explicit flattened participant count.
/// This preserves the legacy 1-D logical grid behavior.
/// \param nspCount  Number of logical participants in the shard grid.
/// \param shardingAllowCollectives  If true, the planner may select sharding
///        plans that require cross-NSP communication (e.g. all-reduce for
///        reduction dimensions). Currently not implemented downstream.
std::unique_ptr<mlir::Pass>
createNSPShardPlannerPass(int64_t nspCount, bool shardingAllowCollectives);

/// Create the planner pass with an explicit 2-D hardware grid.
/// \param numCores  Number of NSP cores in grid axis 0.
/// \param numThreads  Number of threads per core in grid axis 1.
/// \param shardingAllowCollectives  If true, the planner may select sharding
///        plans that require cross-NSP communication.
std::unique_ptr<mlir::Pass>
createNSPShardPlannerPass(int64_t numCores, int64_t numThreads,
                          bool shardingAllowCollectives);

/// Create the NSP sharding propagation pass. This is a thin wrapper around
/// upstream MLIR's ShardingPropagation that additionally treats
/// bufferization.materialize_in_destination as a hard propagation boundary
/// (preventing sharding from crossing a tensor-memref transition).
std::unique_ptr<mlir::Pass> createNSPShardingPropagationPass();

/// Create the NSP localization pass (default: no collectives).
/// Rewrites shard-annotated linalg.generic ops into local tensor compute and
/// emits nsp.materialize_tile for the destination hand-off. In non-collective
/// mode, also distributes scf.for loops cyclically across participants.
std::unique_ptr<mlir::Pass> createNSPLocalizePass();

/// Create the localization pass with explicit collective policy
std::unique_ptr<mlir::Pass>
createNSPLocalizePass(bool shardingAllowCollectives);

/// Create the NSP materialization pass. Consumes nsp-materialize tile ops
/// and rewrites them into per-participant memref.subview stores, using
/// shard.process_linear_index to compute tile offsets.
std::unique_ptr<mlir::Pass> createNSPMaterializePass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Register individual NSP passes with the MLIR pass registry so they are
/// visible to --list-passes and can be invoked standalone (e.g. for lit tests):
///   -nsp-shard-planner, -nsp-sharding-propagation,
///   -nsp-localize, -nsp-materialize
void registerNSPPasses();

/// Register the composite "nsp-shard" pipeline with the MLIR pipeline registry.
/// Invoked as: --pass-pipeline="nsp-shard(nsp-count=32 propagate=true ...)"
void registerNSPPipelines();

/// Register NSP-related dialect extensions (shard, tensor, bufferization, etc.)
/// into the given DialectRegistry. Must be called before context creation so
/// that ShardingInterface models are available during propagation.
void registerNSPDialectExtensions(mlir::DialectRegistry &registry);

/// Register external ShardingInterface models for ops that participate in shard
/// propagation, but are not part of the shard dialect (e.g., linalg.generic,
/// tensor.expand_shape, bufferization.alloc_tensor, memref ops).
void registerNSPShardInterfaceModels(mlir::DialectRegistry &registry);

} // namespace hexagon
} // namespace mlir

#endif // QCOM_HEXAGON_BACKEND_NSP_NSPPASSES_H
