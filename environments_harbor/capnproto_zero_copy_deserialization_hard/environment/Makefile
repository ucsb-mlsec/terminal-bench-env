CXX = g++
CXXFLAGS = -std=c++11 -O2
CAPNP_FLAGS = `pkg-config --cflags --libs capnp`

all: benchmark

message.capnp.c++ message.capnp.h: message.capnp
	capnp compile -oc++ message.capnp

benchmark: benchmark.cpp message.capnp.c++ message.capnp.h
	$(CXX) $(CXXFLAGS) -o benchmark benchmark.cpp message.capnp.c++ $(CAPNP_FLAGS)

clean:
	rm -f benchmark message.capnp.c++ message.capnp.h

.PHONY: all clean