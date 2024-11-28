include ../Makefile.in
include ../common.mk

CXXFLAGS = $(TACS_OPT_CC_FLAGS)
INCLUDE_FLAGS := $(A2D_INCLUDE) $(GPU_MACRO_INCLUDE)
LD_FLAGS :=

ifeq ($(GPU), true)
	CXX := ${GPU_CXX}
	CXXFLAGS := ${GPU_CC_FLAGS} -Xcompiler "${CXXFLAGS}"
endif

.PHONY: benchmark
benchmark: LD_FLAGS += $(BENCHMARK_LINK)
benchmark: INCLUDE_FLAGS += ${BENCHMARK_INCLUDE}
benchmark: LagrangePolyBenchmark.exe

.PHONY: gpuBenchmark
gpuBenchmark: LD_FLAGS += $(BENCHMARK_LINK)
gpuBenchmark: INCLUDE_FLAGS += ${BENCHMARK_INCLUDE}
gpuBenchmark: LagrangePolyGPUBenchmark.exe

%.exe: %.cpp $(deps)
	$(CXX) ${CXX_STD}  $< $(CXXFLAGS) ${INCLUDE_FLAGS} ${LD_FLAGS} -o $@
	@echo
	@echo "        --- Successfully compiled $*.cpp to $*.exe ---"
	@echo

.PHONY: clean
clean:
	rm -f *.exe *.o
