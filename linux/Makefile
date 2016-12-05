# map environment variables to internal variables
XCLASSDIR=${XCLASS_DIR}
PHYSXDIR=${PHYSX_DIR}

XGLDIR=${XCLASSDIR}/xgl
XALDIR=${XCLASSDIR}/xal
XAVDIR=${XCLASSDIR}/xav
EXAMPLESDIR=${XCLASSDIR}/../examples
THIRDPARTYDIR=${XCLASSDIR}/3rdParty

LIB=libxclass.a

CXXFLAGS +=-std=c++14\
	-I${XALDIR} \
	-I${XAVDIR} \
	-I${XGLDIR} \
	-I${XGLDIR}/glm \
	-I${XGLDIR}/glm/gtc \
	-I${THIRDPARTYDIR}/soil/src \
	-I${THIRDPARTYDIR}/freetype/include/freetype2 \
	-I${THIRDPARTYDIR}/anttweakbar/include \
	-I${XCLASSDIR} \
	-I${EXAMPLESDIR}/common \
	-I${EXAMPLESDIR}/PhysXTest \
	-I/usr/include/AL \
	-I${PHYSXDIR}/Include \
	-DNDEBUG \
	-D_GNU_SOURCE=1 \
	-DGLEW_STATIC \
	-Wno-deprecated-declarations \
	-g

CFLAGS +=-std=c11 \
	-I${XALDIR} \
	-I${XAVDIR} \
	-I${XGLDIR} \
	-I${XGLDIR}/glm \
	-I${XCLASSDIR} \
	-D_GNU_SOURCE=1 \
	-DGLEW_STATIC \
	-g

CPPSOURCES=${XGLDIR}/xgl.cpp \
	${XGLDIR}/xglbuffer.cpp \
	${XGLDIR}/xglcamera.cpp \
	${XGLDIR}/xglfont.cpp \
	${XGLDIR}/xglframebuffer.cpp \
	${XGLDIR}/xgllight.cpp \
	${XGLDIR}/xglmaterial.cpp \
	${XGLDIR}/xglobject.cpp \
	${XGLDIR}/xglprojector.cpp \
	${XGLDIR}/xglshader.cpp \
	${XGLDIR}/xglshapes.cpp \
	${XGLDIR}/xgluniform.cpp \
	${XGLDIR}/xglworldcursor.cpp \
	${XALDIR}/xal.cpp \
	${XAVDIR}/xav.cpp \
	${XAVDIR}/xavenc.cpp \
	${XAVDIR}/xavsrc.cpp \
	${XCLASSDIR}/xassets.cpp \
	${XCLASSDIR}/xinput.cpp \
	${XCLASSDIR}/JSON.cpp \
	${XCLASSDIR}/JSONValue.cpp \
	${XCLASSDIR}/xshmem.cpp \


CSOURCES=${XGLDIR}/glew.c \
	${XCLASSDIR}/xutils.c \
	${THIRDPARTYDIR}/soil/src/image_DXT.c \
	${THIRDPARTYDIR}/soil/src/image_helper.c \
	${THIRDPARTYDIR}/soil/src/SOIL.c \
	${THIRDPARTYDIR}/soil/src/stb_image_aug.c \


#COMMON=main.cpp ${EXAMPLESDIR}/common/ExampleXGL.cpp ${EXAMPLESDIR}/common/ExampleGUI.cpp
COMMON=${EXAMPLESDIR}/common/main.cpp ${EXAMPLESDIR}/common/ExampleXGL.cpp ${EXAMPLESDIR}/common/ExampleGUI.cpp

OBJS = $(patsubst %.cpp,%.o,${CPPSOURCES})
OBJS += $(patsubst %.c,%.o,${CSOURCES})

COMMONOBJS = $(patsubst %.cpp,%.o,${COMMON})

LDFLAGS +=\
	-L. \
	-L${THIRDPARTYDIR}/glfw/lib \
	-L${THIRDPARTYDIR}/anttweakbar/lib \
	-L${PHYSXDIR}/Lib/linux64 \
	-L${PHYSXDIR}/Bin/linux64 \
	-L/usr/lib/x86_64-linux-gnu

LIBS=-lxclass -lglfw3 -lGLU -lGL -lX11 -ldl -lXxf86vm -lX11 -lXrandr -lXi -lXinerama -lXcursor -lpthread -lexpat -lfreetype -lopencv_core -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lavformat -lavcodec -lavutil

OPENCV_LIBS=-lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_objdetect -lopencv_ocl -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab

OPENAL_LIBS=-lopenal

PHYSX_LIBS=-lPhysX3Extensions -lPhysX3Common_x64 -lPhysX3_x64 -lPhysX3Cooking_x64 -lPhysX3CharacterKinematic_x64 -lPhysX3Vehicle -lPhysXProfileSDK -lPhysXVisualDebuggerSDK -lPxTask

all: Example02 OpenCVTest OpenCVWebcam OpenALTest PhysXTest

PhysXTest: ${LIB} ${EXAMPLESDIR}/common/ExampleGUI.o ${EXAMPLESDIR}/common/ExampleXGL.o physx-main.o ${EXAMPLESDIR}/PhysXTest/PhysXTestBuildScene.o ${EXAMPLESDIR}/PhysXTest/physx-xgl.o
	${CXX} ${CXXFLAGS} $? ${LDFLAGS} ${LIBS} ${PHYSX_LIBS} -o $@

OpenCVWebcam: ${COMMONOBJS} ${LIB} ${EXAMPLESDIR}/OpenCVWebcam/OpenCVWebcamBuildScene.o
	${CXX} ${CXXFLAGS} $? ${LDFLAGS} ${LIBS} ${OPENCV_LIBS} -o $@

OpenCVTest: ${COMMONOBJS} ${LIB} ${EXAMPLESDIR}/OpenCVTest/OpenCVTestBuildScene.o
	${CXX} ${CXXFLAGS} $? ${LDFLAGS} ${LIBS} ${OPENCV_LIBS} -o $@

OpenALTest: ${COMMONOBJS} ${LIB} ${EXAMPLESDIR}/OpenALTest/OpenALTestBuildScene.o
	${CXX} ${CXXFLAGS} $? ${LDFLAGS} ${LIBS} ${OPENCV_LIBS} ${OPENAL_LIBS} -o $@

Example02: ${COMMONOBJS} ${LIB} ${EXAMPLESDIR}/Example02/Example02.o
	${CXX} ${CXXFLAGS} $? ${LDFLAGS} ${LIBS} -o $@

XAVTest: ${COMMONOBJS} ${LIB} ${EXAMPLESDIR}/XAVTest/XAVTestBuildScene.o
	${CXX} ${CXXFLAGS} $? ${LDFLAGS} ${LIBS} ${OPENAL_LIBS} -o $@

${LIB}: ${OBJS}
	${AR} rcs ${LIB} $? 

clean_exe:
	@-rm -rf Example*
	@-rm -rf Open*
	@-rm -rf *Test

clean:
	@-rm -rf Example*
	@-rm -rf Open*
	@-rm -rf *Test
	@-rm -rf ${COMMONOBJS}
	@-rm -rf ${OBJS}
	@-rm -rf ${LIB}
	@-find .. -name \*.o -exec rm -rf {} \;

