include ../Makefile.in

CXXFLAGS = -std=c++17 -fopenmp #-Wall -Wextra -pedantic -Wno-unused-parameter
CXXOPTFLAGS = -Ofast -march=native -mtune=native
CXXDEBUGFLAGS = -g

obj = TACSHelpers.o
exe = RunTACS.exe

# Add header dependencies
deps = TACSHelpers.h ../element/ResidualKernelPrototype.h

.PHONY: default
default: CXXFLAGS += $(CXXOPTFLAGS)
default: $(exe)

.PHONY: debug
debug: CXXFLAGS += $(CXXDEBUGFLAGS)
debug: $(exe)

%.exe: %.cpp $(obj) $(deps)
	$(CXX) $(CXXFLAGS) $(TACS_FLAGS) $(A2D_INCLUDE) $(GPU_MACRO_INCLUDE) $(TACS_LD_FLAGS) $(obj) $< -o $@

%.o: %.cpp %.h
	$(CXX) $(CXXFLAGS) $(TACS_FLAGS) $(A2D_INCLUDE) $(GPU_MACRO_INCLUDE) -c $< -o $@

.PHONY: clean
clean:
	rm -f *.exe *.o
