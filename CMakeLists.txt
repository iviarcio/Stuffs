add_mlir_library(MLIRNSPTransforms
  NSPLocalizePass.cpp
  NSPMaterializePass.cpp
  NSPPasses.cpp
  NSPShardInterfaceModels.cpp
  NSPShardPlanner.cpp
  NSPShardingPropagationPass.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/hexagon/NSP
    ${PROJECT_SOURCE_DIR}/include/hexagon/Dialect/NSP/IR

  DEPENDS
    NSPTableGen
    # MLIRNSPDialectIncGen
    # MLIRNSPOpsIncGen

  LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRBufferizationDialect
    MLIRFuncDialect
    MLIRIR
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRMemRefDialect
    MLIRNSP
    MLIRPass
    MLIRSCFDialect
    MLIRShardDialect
    MLIRShardTransforms
    MLIRTensorDialect
    MLIRTransformUtils
    MLIRTransforms
)
