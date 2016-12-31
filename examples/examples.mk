ifndef TOPDIR
include ../toplevel.mk
endif

# map environment variables to internal variables
export

EXAMPLESDIR=${TOPDIR}examples

CXXFLAGS += -I${EXAMPLESDIR} -I${EXAMPLESDIR}/common

PROGRAM_SOURCES += ${EXAMPLESDIR}/common/main.cpp \
	${EXAMPLESDIR}/common/ExampleXGL.cpp \
	${EXAMPLESDIR}/common/ExampleGUI.cpp
