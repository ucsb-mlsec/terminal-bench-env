CXX = g++
CXXFLAGS = -O3 -std=c++17 -Wall

TARGETS = model1 model2 model3

.PHONY: all clean

all: $(TARGETS)

model1: model1.cpp
	$(CXX) $(CXXFLAGS) -o model1 model1.cpp

model2: model2.cpp
	$(CXX) $(CXXFLAGS) -o model2 model2.cpp

model3: model3.cpp
	$(CXX) $(CXXFLAGS) -o model3 model3.cpp

clean:
	rm -f $(TARGETS)