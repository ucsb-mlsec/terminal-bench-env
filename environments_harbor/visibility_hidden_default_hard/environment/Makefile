# Makefile for building shared libraries with symbol visibility control

CXX = g++
CXXFLAGS = -fPIC -Wall -std=c++11
LDFLAGS = -shared

# Library names
LIBCORE = libcore.so
LIBUTILS = libutils.so
LIBPLUGIN = libplugin.so

# Source files
CORE_SRC = core.cpp
UTILS_SRC = utils.cpp
PLUGIN_SRC = plugin.cpp

# Test programs
TEST_PROGS = test_core test_utils test_plugin

.PHONY: all clean test

all: $(LIBCORE) $(LIBUTILS) $(LIBPLUGIN)

$(LIBCORE): $(CORE_SRC)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

$(LIBUTILS): $(UTILS_SRC)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

$(LIBPLUGIN): $(PLUGIN_SRC) $(LIBCORE) $(LIBUTILS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $< -L. -lcore -lutils -Wl,-rpath,'$$ORIGIN'

test_core: test_core.cpp $(LIBCORE)
	$(CXX) $(CXXFLAGS) -o $@ $< -L. -lcore -Wl,-rpath,'$$ORIGIN'

test_utils: test_utils.cpp $(LIBUTILS)
	$(CXX) $(CXXFLAGS) -o $@ $< -L. -lutils -Wl,-rpath,'$$ORIGIN'

test_plugin: test_plugin.cpp $(LIBPLUGIN) $(LIBCORE) $(LIBUTILS)
	$(CXX) $(CXXFLAGS) -o $@ $< -L. -lplugin -lcore -lutils -Wl,-rpath,'$$ORIGIN'

test: $(TEST_PROGS)
	@echo "Running test_core..."
	./test_core
	@echo "Running test_utils..."
	./test_utils
	@echo "Running test_plugin..."
	./test_plugin
	@echo "All tests passed!"

clean:
	rm -f $(LIBCORE) $(LIBUTILS) $(LIBPLUGIN) $(TEST_PROGS) *.o