obj-m := module_a_driver.o

KDIR := /lib/modules/5.15.0-91-generic/build

PWD := $(shell pwd)

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean