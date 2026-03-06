Embedded Linux Build System
================================

OVERVIEW
--------
This is a simplified embedded Linux build system that mimics the structure and
workflow of popular build frameworks like Buildroot. It provides infrastructure
for integrating custom applications and libraries into a minimal embedded Linux
root filesystem.

The system uses a package-based architecture where each software component is
defined through standardized configuration and build definition files. A build
orchestrator validates the package definitions and generates manifests for the
target system.


DIRECTORY STRUCTURE
-------------------
The workspace is organized as follows:

/workspace/buildsys/
    Main build system directory containing all infrastructure files and
    package definitions.

/workspace/buildsys/package/
    Package definitions directory. Each package should have its own
    subdirectory containing configuration and makefile fragments.

/workspace/buildsys/configs/
    System configuration files (defconfig files) that define which packages
    and features are enabled for specific target systems.

/workspace/buildsys/Config.in
    Master configuration file that aggregates all package configuration
    options. Packages must be sourced here to appear in the build system.

/workspace/app-source/
    Source code for custom applications. Contains monitor.c which is a
    system monitoring application that needs to be integrated.


PACKAGE INTEGRATION WORKFLOW
-----------------------------
To integrate a new application or library into the build system:

1. Create a package subdirectory under package/<package-name>/

2. Create Config.in file in the package directory:
   - Defines configuration options using Kconfig syntax
   - Specifies package name, help text, and dependencies
   - Gets sourced by the main Config.in file

3. Create <package-name>.mk file in the package directory:
   - Defines how to build the package
   - Specifies source location and build commands
   - Defines installation rules and target paths
   - Declares any dependencies on other packages

4. Register the package in /workspace/buildsys/Config.in:
   - Add a source directive pointing to the package's Config.in

5. Enable the package in the appropriate defconfig file:
   - Add BR2_PACKAGE_<PACKAGE_NAME>=y to enable it

6. Run validate_build.sh to verify the integration:
   - Checks all package definitions are complete
   - Validates configuration syntax
   - Ensures dependencies can be resolved
   - Verifies installation paths are specified


CURRENT STATE
-------------
A monitoring application has been developed and its source code exists at:
    /workspace/app-source/monitor.c

This application provides system resource monitoring capabilities and needs to
be integrated into the build system so it can be included in target rootfs
images.

ISSUES IDENTIFIED:
- No package directory exists for the monitor application
- Package definition files (Config.in and monitor.mk) are missing
- The main Config.in doesn't reference the monitor package
- The defconfig file doesn't enable the monitor package
- Build orchestrator cannot validate the package integration

The validate_build.sh script currently fails due to these missing components.


TASK OBJECTIVE
--------------
Your goal is to properly integrate the monitoring application into the build
system by:

1. Creating the required package directory structure
2. Writing appropriate package definition files
3. Registering the package in the build configuration
4. Enabling the package in the system configuration
5. Ensuring all paths and dependencies are correctly specified

When complete, the validate_build.sh script should run successfully and the
monitoring application should be properly configured for installation in the
target root filesystem.


VALIDATION AND RESULTS
-----------------------
After completing the integration, run:
    /workspace/buildsys/validate_build.sh

This will check:
- Package structure completeness
- Configuration file syntax and references
- Dependency resolution
- Installation path specifications
- Overall build system consistency

Your solution must produce a result file at:
    /tmp/integration_result.txt

This file should contain three lines indicating:
- VALIDATION_STATUS: Whether validation passed or failed
- PACKAGE_REGISTERED: Whether the package is properly registered
- INSTALL_PATH: The full target path where the binary will be installed


HINTS
-----
- Examine existing package structures if any exist in package/ directory
- Config.in files use Kconfig syntax with 'config BR2_PACKAGE_*' directives
- Makefile fragments (.mk files) define build variables and rules
- Standard installation paths are /usr/bin, /usr/sbin, /bin, or /sbin
- Package names typically use lowercase with hyphens
- Configuration symbols use uppercase with underscores
- The source command in Config.in uses relative paths from buildsys/
- Monitor application is a C program requiring compilation with gcc