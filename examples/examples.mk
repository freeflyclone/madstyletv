ifndef TOPDIR
include ../toplevel.mk
endif

# map environment variables to internal variables
export

EXAMPLESDIR=${TOPDIR}examples

CXXFLAGS += -I${EXAMPLESDIR} -I${EXAMPLESDIR}/common

EXAMPLESCOMMON=${EXAMPLESDIR}/common/main.cpp ${EXAMPLESDIR}/common/ExampleXGL.cpp ${EXAMPLESDIR}/common/ExampleGUI.cpp
EXAMPLESCOMMONOBJS = $(patsubst %.cpp,%.o,${EXAMPLESCOMMON})

