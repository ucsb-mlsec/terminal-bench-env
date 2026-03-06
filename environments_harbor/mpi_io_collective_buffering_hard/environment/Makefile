CC = mpicc
CFLAGS = -O2 -Wall

all: climate_sim

climate_sim: climate_sim.c
	$(CC) $(CFLAGS) -o climate_sim climate_sim.c

clean:
	rm -f climate_sim checkpoint.dat