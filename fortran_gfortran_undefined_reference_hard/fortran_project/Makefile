FC = gfortran
FFLAGS = -c
SRCDIR = src

all: science_app

science_app: main.o module_compute.o module_math.o module_constants.o
	$(FC) -o science_app main.o module_compute.o module_math.o module_constants.o

main.o: $(SRCDIR)/main.f90
	$(FC) $(FFLAGS) $(SRCDIR)/main.f90

module_compute.o: $(SRCDIR)/module_compute.f90
	$(FC) $(FFLAGS) $(SRCDIR)/module_compute.f90

module_math.o: $(SRCDIR)/module_math.f90
	$(FC) $(FFLAGS) $(SRCDIR)/module_math.f90

module_constants.o: $(SRCDIR)/module_constants.f90
	$(FC) $(FFLAGS) $(SRCDIR)/module_constants.f90

clean:
	rm -f *.o science_app