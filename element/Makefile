include ../Makefile.in

CXX = g++
CXXFLAGS = -std=c++17 # -Wall -Wextra -pedantic
CXXOPTFLAGS = -Ofast -march=native -mtune=native
CXXDEBUGFLAGS = -g
LINK_FLAGS =


.PHONY: benchmark
default: CXXFLAGS += $(CXXOPTFLAGS)
default: LINK_FLAGS += $(BENCHMARK_LINK)
default: LagrangePolyBenchmark.exe

%.exe: %.cpp
	$(CXX) $< $(CXXFLAGS) $(A2D_INCLUDE) $(LINK_FLAGS) -o $@

.PHONY: clean
clean:
	rm -f *.exe *.o
