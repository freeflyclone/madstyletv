include ../../xclass/xclass.mk

PHYSXDIR=${PHYSX_DIR}
EXAMPLESDIR=${XCLASSDIR}/../examples

CXXFLAGS+=-I${PHYSX_DIR}/Include\
	-I${EXAMPLESDIR}/common\
	-I.

LDFLAGS+=-L${PHYSX_DIR}/Lib/linux64\
	-L${PHYSX_DIR}/Bin/linux64\


PROGRAM=PhysXTest

CPPSOURCES=PhysXTestBuildScene.cpp\
	physx-main.cpp\
	physx-xgl.cpp\
	${EXAMPLESDIR}/common/ExampleXGL.cpp\
	${EXAMPLESDIR}/common/ExampleGUI.cpp\


OBJS = $(patsubst %.cpp,%.o,${CPPSOURCES})

PHYSX_LIBS=-lPhysX3Extensions -lPhysX3Common_x64 -lPhysX3_x64 -lPhysX3Cooking_x64 -lPhysX3CharacterKinematic_x64 -lPhysX3Vehicle -lPhysXProfileSDK -lPhysXVisualDebuggerSDK -lPxTask

#QtCreator likes an 'all' target

all: ${PROGRAM}

${PROGRAM}: ${OBJS} ${LIB}
	${CXX} ${CPPFLAGS} $? ${LDFLAGS} ${LIBS} ${PHYSX_LIBS} -o $@

clean:
	@-rm -rf ${PROGRAM}
	@-rm -rf ${OBJS}

