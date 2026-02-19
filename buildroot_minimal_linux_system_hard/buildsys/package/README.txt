Package Directory Structure
=========================

This directory contains package definitions for the build system. Each package
that can be built and included in the final system image must have its own
subdirectory under this package/ directory.

Required Package Files
----------------------

Each package subdirectory must contain at least two files:

1. Config.in - Defines configuration options for the package
2. <packagename>.mk - Contains build and installation instructions

The package name should be lowercase and match the subdirectory name.

Config.in File Format
---------------------

The Config.in file defines how the package appears in the build configuration.
It must follow this structure:

- Define a configuration option using: config BR2_PACKAGE_PACKAGENAME
- The package name in the config should be uppercase
- Specify the type (typically 'bool') followed by a description string
- Optionally include a help section with detailed information
- Dependencies can be specified with 'depends on' statements

Example Config.in:
    config BR2_PACKAGE_MYAPP
        bool "myapp - My custom application"
        help
          This is my custom application that does something useful.
          It will be installed to /usr/bin/myapp in the target system.

Makefile (.mk) File Format
--------------------------

The package makefile defines how to build and install the package. It must
define several variables using the package name in uppercase:

Required variables:
- PACKAGENAME_VERSION: Version string for the package
- PACKAGENAME_SOURCE: Path to the source code (can be local path)
- PACKAGENAME_INSTALL_TARGET_CMDS: Commands to install files to target

Optional variables:
- PACKAGENAME_BUILD_CMDS: Commands to build the package
- PACKAGENAME_DEPENDENCIES: Other packages this one depends on

The makefile must end with a call to the package infrastructure:
    $(eval $(generic-package))

Example myapp.mk:
    MYAPP_VERSION = 1.0
    MYAPP_SOURCE = /workspace/app-source/myapp.c
    
    define MYAPP_BUILD_CMDS
        $(TARGET_CC) $(TARGET_CFLAGS) -o $(@D)/myapp $(MYAPP_SOURCE)
    endef
    
    define MYAPP_INSTALL_TARGET_CMDS
        $(INSTALL) -D -m 0755 $(@D)/myapp $(TARGET_DIR)/usr/bin/myapp
    endef
    
    $(eval $(generic-package))

Complete Example Package
-------------------------

For a package called "monitor", create the following structure:

Directory: package/monitor/

File: package/monitor/Config.in
    config BR2_PACKAGE_MONITOR
        bool "monitor - System monitoring application"
        help
          A custom monitoring application for system metrics.
          Installs to /usr/bin/monitor on the target system.

File: package/monitor/monitor.mk
    MONITOR_VERSION = 1.0
    MONITOR_SOURCE = /workspace/app-source/monitor.c
    
    define MONITOR_BUILD_CMDS
        $(TARGET_CC) $(TARGET_CFLAGS) -o $(@D)/monitor $(MONITOR_SOURCE)
    endef
    
    define MONITOR_INSTALL_TARGET_CMDS
        $(INSTALL) -D -m 0755 $(@D)/monitor $(TARGET_DIR)/usr/bin/monitor
    endef
    
    $(eval $(generic-package))

Enabling Packages
-----------------

After creating the package files:

1. The package must be referenced in the main Config.in file at the root
   of the buildsys directory using: source "package/packagename/Config.in"

2. To include the package in a build configuration, add it to the appropriate
   defconfig file in configs/ directory with: BR2_PACKAGE_PACKAGENAME=y

3. Run the build validation to ensure everything is configured correctly

Notes
-----

- All package names should be lowercase in directory names and filenames
- Config option names (BR2_PACKAGE_*) should be uppercase
- The $(TARGET_DIR) variable points to the root filesystem staging area
- The $(@D) variable in makefiles refers to the package build directory
- Installation commands should use $(INSTALL) for proper permissions