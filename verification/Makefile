include ../Makefile.in
include ../common.mk

CXXFLAGS = -std=c++17 #-fopenmp -Wall -Wextra -pedantic -Wno-unused-parameter

obj = TACSHelpers.o
exe = RunTACS.exe

# Add header dependencies
deps = TACSHelpers.h ../element/ResidualKernelPrototype.h

.PHONY: default
default: CXXFLAGS := $(TACS_OPT_CC_FLAGS)
default: $(exe)

.PHONY: debug
debug: CXXFLAGS := $(TACS_DEBUG_CC_FLAGS)
debug: $(exe)

%.exe: %.cpp $(obj) $(deps)
	$(CXX) $(CXXFLAGS) $(TACS_FLAGS) ${MPI_INC} $(A2D_INCLUDE) $(GPU_MACRO_INCLUDE) ${TACS_INCLUDE} $(obj) ${TACS_LD_FLAGS} $< -o $@

TACSHelpers.o: TACSHelpers.cpp TACSHelpers.h
	$(CXX) $(TACS_OPT_CC_FLAGS) $(A2D_INCLUDE) $(GPU_MACRO_INCLUDE) ${TACS_INCLUDE} -c TACSHelpers.cpp -o TACSHelpers.o

.PHONY: clean
clean:
	rm -f *.exe *.o
