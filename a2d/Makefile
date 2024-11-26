include ../Makefile.in

CXX = g++
CXXFLAGS = -std=c++17 # -Wall -Wextra -pedantic
CXXOPTFLAGS = -Ofast -march=native -mtune=native
CXXDEBUGFLAGS = -g

.PHONY: default
default: CXXFLAGS += $(CXXOPTFLAGS)
default: a2d-learning.exe

.PHONY: debug
debug: CXXFLAGS += $(CXXDEBUGFLAGS)
debug: a2d-learning.exe

.PHONY: gpu
gpu:
	nvcc -std=c++17 -I/home6/achris10/repos/a2d/include  -I/home6/achris10/repos/GPU-FEM-Project  a2d-learning.cu -o a2d-learning.exe

%.exe: %.cpp
	$(CXX) $(CXXFLAGS) $(A2D_INCLUDE) ${GPU_INCLUDE} $< -o $@

.PHONY: clean
clean:
	rm -f *.exe *.o
