# map environment variables to internal variables
export

EXAMPLESDIR=${XCLASSDIR}/../examples

CXXFLAGS += -I${EXAMPLESDIR} -I${EXAMPLESDIR}/common

COMMON=${EXAMPLESDIR}/common/main.cpp ${EXAMPLESDIR}/common/ExampleXGL.cpp ${EXAMPLESDIR}/common/ExampleGUI.cpp
COMMONOBJS = $(patsubst %.cpp,%.o,${COMMON})

