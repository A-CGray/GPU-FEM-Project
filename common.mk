TACS_LIB = ${TACS_DIR}/lib/liba2dshells.a

TACS_INCLUDE = -I${TACS_DIR}/src \
	-I${TACS_DIR}/src/bpmat \
	-I${TACS_DIR}/src/elements \
	-I${TACS_DIR}/src/elements/a2d \
	-I${TACS_DIR}/src/elements/dynamics \
	-I${TACS_DIR}/src/elements/basis \
	-I${TACS_DIR}/src/elements/shell \
	-I${TACS_DIR}/src/constitutive \
	-I${TACS_DIR}/src/functions \
	-I${TACS_DIR}/src/io

# Set the command line flags to use for compilation
TACS_INCLUDE := ${METIS_INCLUDE} ${AMD_INCLUDE} ${TECIO_INCLUDE} ${A2D_INCLUDE} ${TACS_INCLUDE}

TACS_OPT_CC_FLAGS = ${CXX_STD} ${EXTRA_CC_FLAGS}
TACS_DEBUG_CC_FLAGS = ${CXX_STD} ${EXTRA_DEBUG_CC_FLAGS}

# Set the linking flags to use
TACS_EXTERN_LIBS = ${AMD_LIBS} ${METIS_LIB} ${LAPACK_LIBS} ${TECIO_LIBS}
TACS_LD_FLAGS = ${EXTRA_LD_FLAGS} ${TACS_LD_CMD} ${TACS_EXTERN_LIBS}
GPU_TACS_LD_FLAGS = ${EXTRA_LD_FLAGS} ${GPU_TACS_LD_CMD} ${TACS_EXTERN_LIBS} ${MPI_LIB} ${CUDA_LIBS}
