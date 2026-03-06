# SPDX-License-Identifier: GPL-2.0
VERSION = 5
PATCHLEVEL = 4
SUBLEVEL = 0
EXTRAVERSION =
NAME = Kleptomaniac Octopus

# *DOCUMENTATION*
# To see a list of typical targets execute "make help"
# More info can be located in ./README
# Comments in this file are targeted only to the developer, do not
# expect to learn how to build the kernel reading this file.

# That's our default target when none is given on the command line
PHONY := _all
_all:

# We are using a recursive build, so we need to do a little thinking
# to get the configuration right.
#
# Most importantly: sub-Makefiles should only ever modify files in
# their own directory. If in some directory we have a dependency on
# a file in another dir (which doesn't happen often, but it's often
# unavoidable when linking the built-in.a targets which finally
# turn into vmlinux), we will call a sub make in that other dir, and
# after that we are sure that everything which is in that other dir
# is now up to date.

# o Do not use make's built-in rules and variables
#   (this increases performance and avoids hard-to-debug behaviour);
# o Look for make include files relative to root of kernel src
MAKEFLAGS += -rR --include-dir=$(CURDIR)

# Avoid funny character set dependencies
unexport LC_ALL
LC_COLLATE=C
LC_NUMERIC=C
export LC_COLLATE LC_NUMERIC

# Avoid interference with shell env settings
unexport GREP_OPTIONS

# We need some generic definitions (do not try to remake the file).
scripts/Kbuild.include: ;
include scripts/Kbuild.include

# Make variables (CC, etc...)
AS		= $(CROSS_COMPILE)as
LD		= $(CROSS_COMPILE)ld
CC		= $(CROSS_COMPILE)gcc
CPP		= $(CC) -E
AR		= $(CROSS_COMPILE)ar
NM		= $(CROSS_COMPILE)nm
STRIP		= $(CROSS_COMPILE)strip
OBJCOPY		= $(CROSS_COMPILE)objcopy
OBJDUMP		= $(CROSS_COMPILE)objdump
LEX		= flex
YACC		= bison
AWK		= awk
INSTALLKERNEL  := installkernel
DEPMOD		= /sbin/depmod
PERL		= perl
PYTHON		= python
PYTHON2		= python2
PYTHON3		= python3
CHECK		= sparse

CHECKFLAGS     := -D__linux__ -Dlinux -D__STDC__ -Dunix -D__unix__ \
		  -Wbitwise -Wno-return-void -Wno-unknown-attribute $(CF)

export VERSION PATCHLEVEL SUBLEVEL EXTRAVERSION LOCALVERSION MODVERDIR KERNELRELEASE

export ARCH SRCARCH CONFIG_SHELL HOSTCC KBUILD_HOSTCFLAGS CROSS_COMPILE AS LD CC
export CPP AR NM STRIP OBJCOPY OBJDUMP HOSTLDFLAGS HOST_LOADLIBES
export MAKE LEX YACC AWK INSTALLKERNEL DEPMOD PERL PYTHON PYTHON2 PYTHON3 UTS_MACHINE
export HOSTCXX KBUILD_HOSTCXXFLAGS LDFLAGS_MODULE CHECK CHECKFLAGS

export KBUILD_CPPFLAGS NOSTDINC_FLAGS LINUXINCLUDE OBJCOPYFLAGS KBUILD_LDFLAGS
export KBUILD_CFLAGS CFLAGS_KERNEL CFLAGS_MODULE
export CFLAGS_KASAN CFLAGS_KASAN_NOSANITIZE CFLAGS_UBSAN
export KBUILD_AFLAGS AFLAGS_KERNEL AFLAGS_MODULE
export KBUILD_AFLAGS_KERNEL KBUILD_CFLAGS_KERNEL
export KBUILD_AFLAGS_MODULE KBUILD_CFLAGS_MODULE

# Files to ignore in find ... statements

export RCS_FIND_IGNORE := \( -name SCCS -o -name BitKeeper -o -name .svn -o    \
			  -name CVS -o -name .pc -o -name .hg -o -name .git \) \
			  -prune -o

# ===========================================================================
# Rules shared between *config targets and build targets

PHONY += scripts_basic
scripts_basic:
	$(Q)$(MAKE) $(build)=scripts/basic
	$(Q)rm -f .tmp_quiet_recordmcount

PHONY += outputmakefile

# Basic helpers built in scripts/basic/
PHONY += scripts_basic
scripts_basic:
	$(Q)$(MAKE) $(build)=scripts/basic

all: modules

PHONY += modules
modules:
	@echo "Building kernel modules"
	$(Q)$(MAKE) -f $(srctree)/scripts/Makefile.build obj=.

clean:
	@echo "Cleaning build artifacts"
	@rm -f *.o *.ko *.mod.* modules.order Module.symvers

help:
	@echo  'Cleaning targets:'
	@echo  '  clean           - Remove most generated files but keep the config'
	@echo  ''
	@echo  'Other generic targets:'
	@echo  '  all             - Build all targets marked with [*]'
	@echo  '* modules         - Build all modules'

.PHONY: $(PHONY)