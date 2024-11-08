include ../Makefile.in

# CXX = mpicxx
CXXFLAGS = -std=c++17 # -Wall -Wextra -pedantic
CXXOPTFLAGS = -Ofast -march=native -mtune=native
CXXDEBUGFLAGS = -g

.PHONY: default
default: CXXFLAGS += $(CXXOPTFLAGS)
default: RunTACS.exe

.PHONY: debug
debug: CXXFLAGS += $(CXXDEBUGFLAGS)
debug: RunTACS.exe

%.exe: %.cpp
	$(CXX) $(CXXFLAGS) $(TACS_FLAGS) $< -o $@

.PHONY: clean
clean:
	rm -f *.exe *.o
