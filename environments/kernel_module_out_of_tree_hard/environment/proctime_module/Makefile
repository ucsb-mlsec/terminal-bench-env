obj-m += proctime.o

KDIR := /lib/modules/5.4.0-generic/build
PWD = $(shell pwd)

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
	rm -f *.o *.ko *.mod.* *.symvers *.order

install:
	$(MAKE) -C $(KDIR) M=$(PWD) modules_install

help:
	@echo "Available targets:"
	@echo "  all     - Build the kernel module"
	@echo "  clean   - Remove build artifacts"
	@echo "  install - Install the module"