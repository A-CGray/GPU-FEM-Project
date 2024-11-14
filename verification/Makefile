include ../Makefile.in

CXXFLAGS = -std=c++17 -Wall -Wextra -pedantic -Wno-unused-parameter
CXXOPTFLAGS = -Ofast -march=native -mtune=native
CXXDEBUGFLAGS = -g

obj = TACSHelpers.o
exe = RunTACS.exe

# Add header dependencies
deps = TACSHelpers.h

.PHONY: default
default: CXXFLAGS += $(CXXOPTFLAGS)
default: $(exe)

.PHONY: debug
debug: CXXFLAGS += $(CXXDEBUGFLAGS)
debug: $(exe)

%.exe: %.cpp $(obj) $(deps)
	$(CXX) $(CXXFLAGS) $(TACS_FLAGS) $(obj) $(TACS_LD_FLAGS) $< -o $@

%.o: %.cpp %.h
	$(CXX) $(CXXFLAGS) $(TACS_FLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f *.exe *.o
