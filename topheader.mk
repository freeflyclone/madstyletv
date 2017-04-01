# Defines variables that help all the subdir Makefiles
# know where to find stuff, as well as project level
# build settings.
#
# Note:
#   The way TOPDIR is created here absolutely depends on
#   this file being the first include in ALL subdir
#   Makefiles, regardless of tree depth, and that each
#   subdir Makefile includes this file with a relative
#   path, which means that the deeper in the source tree
#   a Makefile is, the more "../"es are going to be
#   needed.
#-------------------------------------------------------
.PHONY: clean all

# As it's name might suggest, TOPDIR is the
# top-most directory of the madstyletv source code
# tree.  All subdir Makefiles rely on this
TOPDIR := $(dir $(lastword $(MAKEFILE_LIST)))

# A user-defined gmake function for building a program.
# It also copies it to the project's "bin" folder
PROGRAM_BUILD =$(CXX) -g -o $@ $^ $(LDFLAGS) $(LIBS) ; cp $@ $(TOPDIR)bin

PROGRAM_OBJS = $(patsubst %.cpp,%.o,$(patsubst %.c,%.o,$(PROGRAM_SOURCES)))

ECHO_VARS = @echo PROGRAM_SOURCES: $(PROGRAM_SOURCES) PROGRAM_OBJS: $(PROGRAM_OBJS)

# Change the tools to allow for cross compilation by setting CROSS_COMPILE
# to a toolchain prefix
CXX = $(CROSS_COMPILE)g++
CC = $(CROSS_COMPILE)gcc
AS = $(CROSS_COMPILE)as
AR = $(CROSS_COMPILE)ar
NM = $(CROSS_COMPILE)nm
LD = $(CROSS_COMPILE)ld
OBJDUMP = $(CROSS_COMPILE)objdump
OBJCOPY = $(CROSS_COMPILE)objcopy
RANLIB = $(CROSS_COMPILE)ranlib
STRIP = $(CROSS_COMPILE)strip

