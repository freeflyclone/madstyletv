# map environment variables to internal variables
XCLASSDIR=${XCLASS_DIR}
PHYSXDIR=${PHYSX_DIR}

XGLDIR=${XCLASSDIR}/xgl
XALDIR=${XCLASSDIR}/xal
XAVDIR=${XCLASSDIR}/xav
EXAMPLESDIR=${XCLASSDIR}/../examples
THIRDPARTYDIR=${XCLASSDIR}/3rdParty

PROGRAM=Example02
LIB=${XCLASSDIR}/libxclass.a

CXXFLAGS +=-std=c++14\
	-I${XALDIR} \
	-I${XAVDIR} \
	-I${XGLDIR} \
	-I${XGLDIR}/glm \
	-I${XGLDIR}/glm/gtc \
	-I${THIRDPARTYDIR}/soil/src \
	-I${THIRDPARTYDIR}/freetype/include/freetype2 \
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

COMMON=${EXAMPLESDIR}/common/main.cpp ${EXAMPLESDIR}/common/ExampleXGL.cpp ${EXAMPLESDIR}/common/ExampleGUI.cpp

COMMONOBJS = $(patsubst %.cpp,%.o,${COMMON})

LDFLAGS +=\
	-L${XCLASSDIR} \
	-L${THIRDPARTYDIR}/glfw/lib \
	-L${PHYSXDIR}/Lib/linux64 \
	-L${PHYSXDIR}/Bin/linux64 \
	-L/usr/lib/x86_64-linux-gnu

LIBS=-lxclass -lglfw3 -lGLU -lGL -lX11 -ldl -lXxf86vm -lX11 -lXrandr -lXi -lXinerama -lXcursor -lpthread -lexpat -lfreetype -lopencv_core -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lavformat -lavcodec -lavutil

OPENCV_LIBS=-lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_objdetect -lopencv_ocl -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab

OPENAL_LIBS=-lopenal

PHYSX_LIBS=-lPhysX3Extensions -lPhysX3Common_x64 -lPhysX3_x64 -lPhysX3Cooking_x64 -lPhysX3CharacterKinematic_x64 -lPhysX3Vehicle -lPhysXProfileSDK -lPhysXVisualDebuggerSDK -lPxTask

all: ${PROGRAM}

${PROGRAM}: ${COMMONOBJS} ${LIB} Example02.o
	${CXX} ${CPPFLAGS} $? ${LDFLAGS} ${LIBS} -o $@

clean:
	@-rm -rf ${PROGRAM}
	@-rm -rf ${COMMONOBJS}
