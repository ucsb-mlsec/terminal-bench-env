CXX = g++
CXXFLAGS = -fPIC
LIBDIR = ./lib
SRCDIR = .

all: $(LIBDIR)/libcalculator.so calc_app

$(LIBDIR)/libcalculator.so: $(SRCDIR)/calculator.cpp
	@mkdir -p $(LIBDIR)
	$(CXX) $(CXXFLAGS) -shared -o $(LIBDIR)/libcalculator.so $(SRCDIR)/calculator.cpp

calc_app: $(SRCDIR)/main.cpp $(LIBDIR)/libcalculator.so
	$(CXX) -o calc_app $(SRCDIR)/main.cpp -L$(LIBDIR) -lcalculator

clean:
	rm -f calc_app
	rm -f $(LIBDIR)/libcalculator.so
	rmdir $(LIBDIR) 2>/dev/null || true

.PHONY: all clean