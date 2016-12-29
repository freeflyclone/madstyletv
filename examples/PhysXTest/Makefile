include ../../toplevel.mk
include ../../xclass/xclass.mk
include ../examples.mk

PROGRAM=PhysXTest

PHYSXDIR=${PHYSX_DIR}

CXXFLAGS+=-I${PHYSX_DIR}/Include\
	-I.

LDFLAGS+=-L${PHYSX_DIR}/Lib/linux64\
	-L${PHYSX_DIR}/Bin/linux64\


# Other examples use main.cpp and ExampleXGL.cpp from examples/common
# In this case, we're overriding the ExampleXGL class to include
# PhysX in the scene rendering phase, in a way that's more complex
# than the usual ExampleXGL::BuildScene() technique allows.
CPPSOURCES=PhysXTestBuildScene.cpp\
	physx-main.cpp\
	physx-xgl.cpp\
	${EXAMPLESDIR}/common/ExampleXGL.cpp\
	${EXAMPLESDIR}/common/ExampleGUI.cpp\


OBJS = $(patsubst %.cpp,%.o,${CPPSOURCES})

LIBS+=-lPhysX3Extensions \
	-lPhysX3Common_x64 \
	-lPhysX3_x64 \
	-lPhysX3Cooking_x64 \
	-lPhysX3CharacterKinematic_x64 \
	-lPhysX3Vehicle \
	-lPhysXProfileSDK \
	-lPhysXVisualDebuggerSDK \
	-lPxTask

all: ${PROGRAM}

${PROGRAM}: ${OBJS} ${LIB}
	${CXX} ${CPPFLAGS} $? ${LDFLAGS} ${LIBS} -o $@

clean:
	@-rm -rf ${PROGRAM}
	@-rm -rf ${OBJS}

