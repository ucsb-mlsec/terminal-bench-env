# Custom OpenWrt build definitions

# Package configuration flags - missing quotes around value
CUSTOM_FLAGS:=--enable-feature --with-path=/some/path

# Build output directory reference
BUILD_OUTPUT:=$(NONEXISTENT_BUILDDIR)/output

# Custom package version
CUSTOM_PKG_VERSION:=1.2.3

# Feature toggle with malformed conditional - missing comma
ifeq ($(ENABLE_CUSTOM_FEATURES)yes)
  EXTRA_CFLAGS+=-DCUSTOM_FEATURES
endif

# Line continuation with trailing whitespace after backslash
PACKAGE_LIST:=base-files busybox \  
	kernel-modules network-utils

# Template function with incorrect closing - should be 'endef'
define custom_build_template
	$(1)_BUILD_DIR:=$BUILD_OUTPUT/$(1)
	$(1)_CONFIGURE:=./configure --prefix=/usr
	$(1)_COMPILE:=$(MAKE) -C $$($(1)_BUILD_DIR)
enddef

# Install target with incorrect variable reference
install-custom:
	@echo "Installing to $DESTDIR"
	mkdir -p $(DESTDIR)/usr/lib
	cp -r $(BUILD_OUTPUT)/* $(DESTDIR)/usr/lib/

# Recipe with spaces instead of tab
check-environment:
        @echo "Checking build environment"
        @test -d $(BUILD_OUTPUT) || mkdir -p $(BUILD_OUTPUT)

# Custom architecture detection
CUSTOM_ARCH:=$(shell uname -m | sed 's/x86_64/amd64/g')

# Additional build flags
CFLAGS_EXTRA:=-O2 -Wall -Wextra