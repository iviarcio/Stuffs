//===- NSPPasses.h --------------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Public entry points for registering NSP passes, pipelines and dialect
// extensions required by NSP sharding/planning.
//
// This header is intended to be included by tools/drivers (e.g., *-opt) or by
// the Hexagon backend initialization code that sets up DialectRegistry and
// registers pass pipelines.
//
//===----------------------------------------------------------------------===//

#ifndef QCOM_HEXAGON_BACKEND_NSP_NSPPASSES_H
#define QCOM_HEXAGON_BACKEND_NSP_NSPPASSES_H

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace hexagon {

/// Factory for the NSP shard planner pass.
std::unique_ptr<mlir::Pass> createNSPShardPlannerPass();
std::unique_ptr<mlir::Pass> createNSPShardPlannerPass(int64_t nspCount, bool allowCollectives);

/// NSP-specific sharding propagation wrapper.
std::unique_ptr<mlir::Pass> createNSPShardingPropagationPass();

/// Factories for the NSP Localization/Materialization passes.
std::unique_ptr<mlir::Pass> createNSPLocalizePass();
std::unique_ptr<mlir::Pass> createNSPLocalizePass(bool allowCollectives);

std::unique_ptr<mlir::Pass> createNSPMaterializePass();

/// Register individual NSP passes (e.g., planner, materialization) so they are
/// visible to the MLIR pass registry (e.g., --list-passes).
void registerNSPPasses();

/// Register NSP pass pipelines (e.g., "nsp-shard") so they are visible to the
/// MLIR pass pipeline registry (e.g., --list-pipelines).
void registerNSPPipelines();

/// Register NSP-related dialect extensions and interface models into the given
/// DialectRegistry. This is required so that Shard dialect propagation can
/// reason about non-shard ops that appear in the IR (boundary ops).
///
/// Note: The caller must ensure the returned registry is later used to create
/// the MLIRContext (or is appended into an existing registry bound to a context).
void registerNSPDialectExtensions(mlir::DialectRegistry &registry);

/// Register interface models for operations that need to participate in Shard
/// propagation, but do not live in the Shard dialect (e.g., bufferization and
/// memref boundary ops).
void registerNSPShardInterfaceModels(mlir::DialectRegistry &registry);

} // namespace hexagon
} // namespace mlir

#endif // QCOM_HEXAGON_BACKEND_NSP_NSPPASSES_H
