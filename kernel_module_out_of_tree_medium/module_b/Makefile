obj-m := network_filter.o

KDIR := /usr/src/linux-headers-5.15.0-91-generic

PWD := $(shell pwd)

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean