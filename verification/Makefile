include ../Makefile.in
include ../common.mk

CXXFLAGS = $(TACS_OPT_CC_FLAGS)
ifeq ($(DEBUG), true)
	CXXFLAGS := $(TACS_DEBUG_CC_FLAGS)
	GPU_CC_FLAGS := ${GPU_CC_DEBUG_FLAGS}
endif

obj = TACSHelpers.o
exe = TestKernels.exe

LD_FLAGS := ${TACS_LD_FLAGS}
INCLUDE_FLAGS := $(A2D_INCLUDE) $(GPU_MACRO_INCLUDE) ${TACS_INCLUDE}

ifeq ($(GPU), true)
	LD_FLAGS := ${GPU_TACS_LD_FLAGS}
	CXX := ${GPU_CXX}
	CXXFLAGS := ${GPU_CC_FLAGS} -Xcompiler "${CXX_STD} ${CXXFLAGS}"
endif

# Don't delete the .o files after compiling the executable
.PRECIOUS: $(obj)

# Add header dependencies
deps = TACSHelpers.h ../element/FEKernels.h ../element/LagrangeShapeFuncs.h ../element/GaussQuadrature.h

.PHONY: default
default: $(exe)

%.exe: %.cpp $(deps)
	$(CXX) ${CXX_STD}  $< $(CXXFLAGS) ${INCLUDE_FLAGS} ${LD_FLAGS} -o $@
	@echo
	@echo "        --- Successfully compiled $*.cpp to $*.exe ---"
	@echo

# %.o: %.cpp %.h
# 	$(CXX) ${CXX_STD} $(CXXFLAGS) $(A2D_INCLUDE) $(GPU_MACRO_INCLUDE) ${TACS_INCLUDE} -c $< -o $@
# 	@echo
# 	@echo "        --- Successfully compiled $*.cpp to $*.o ---"
# 	@echo

.PHONY: clean
clean:
	rm -f *.exe *.o
