CXX = g++
CXXFLAGS = -O2 -std=c++11
BOOST_INCLUDE = /usr/include
BOOST_LIB = /usr/lib
TARGET = json_reader

all: $(TARGET)

$(TARGET): main.cpp
	$(CXX) $(CXXFLAGS) -I$(BOOST_INCLUDE) -L$(BOOST_LIB) main.cpp -o $(TARGET) -lboost_system -lboost_filesystem

clean:
	rm -f $(TARGET)

.PHONY: all clean