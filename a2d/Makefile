include ../Makefile.in

CXX=clang++
CXXFLAGS = -std=c++17 # -Wall -Wextra -pedantic
CXXOPTFLAGS = -Ofast -march=native -mtune=native
CXXDEBUGFLAGS = -g

.phony: default
default: $(CXXFLAGS) += $(CXXOPTFLAGS)
default: a2d-learning.exe

.phony: debug
debug: $(CXXFLAGS) += $(CXXDEBUGFLAGS)
debug: a2d-learning.exe

%.exe: %.cpp
	$(CXX) $(CXXFLAGS) $(A2D_INCLUDE) $< -o $@

.phony: clean
clean:
	rm -f *.exe *.o
