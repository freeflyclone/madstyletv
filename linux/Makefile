LIB=libxclass.a
XCLASSDIR=../xclass
XGLDIR=../xclass/xgl

CPPFLAGS +=-std=c++14\
	-I../xclass/xgl \
	-I../xclass/xgl/glm \
	-I../xclass/xgl/glm/gtc \
	-I../xclass/3rdParty/soil/src \
	-I../xclass/3rdParty/freetype/include/freetype2 \
	-I../xclass \
	-I../examples/common \
	-I/usr/include/AL
	-D_GNU_SOURCE=1 \
	-DGLEW_STATIC \
	-Wno-writable-strings \
	-Wno-deprecated-declarations \
	-g

CFLAGS +=-std=c11 \
	-Iglm \
	-I../xclass/xgl \
	-I../xclass/xgl/glm \
	-I../xclass \
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
	${XCLASSDIR}/xinput.cpp \
	${XCLASSDIR}/xconfig.cpp \
	${XCLASSDIR}/xshmem.cpp \


CSOURCES=${XGLDIR}/glew.c \
	${XCLASSDIR}/xutils.c \
	${XCLASSDIR}/3rdParty/soil/src/image_DXT.c \
	${XCLASSDIR}/3rdParty/soil/src/image_helper.c \
	${XCLASSDIR}/3rdParty/soil/src/SOIL.c \
	${XCLASSDIR}/3rdParty/soil/src/stb_image_aug.c \


COMMON=main.cpp ../examples/common/ExampleXGL.cpp

OBJS = $(patsubst %.cpp,%.o,${CPPSOURCES})
OBJS += $(patsubst %.c,%.o,${CSOURCES})

COMMONOBJS = $(patsubst %.cpp,%.o,${COMMON})

LDFLAGS +=\
	-L. \
	-L../xclass/3rdParty/glfw/lib \
	-L/usr/lib/x86_64-linux-gnu

LIBS=-lxclass -lglfw3 -lGLU -lGL -lX11 -ldl -lXxf86vm -lX11 -lXrandr -lXi -lXinerama -lXcursor -lpthread -lexpat -lfreetype -lopencv_core -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d 

OPENCV_LIBS=-lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_objdetect -lopencv_ocl -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab

OPENAL_LIBS=-lopenal

all: Example02 OpenCVTest OpenCVWebcam OpenALTest

OpenCVWebcam: ${COMMONOBJS} ${LIB} ../examples/OpenCVWebcam/OpenCVWebcamBuildScene.o
	${CXX} ${CPPFLAGS} $? ${LDFLAGS} ${LIBS} ${OPENCV_LIBS} -o $@

OpenCVTest: ${COMMONOBJS} ${LIB} ../examples/OpenCVTest/OpenCVTestBuildScene.o
	${CXX} ${CPPFLAGS} $? ${LDFLAGS} ${LIBS} ${OPENCV_LIBS} -o $@

OpenALTest: ${COMMONOBJS} ${LIB} ../examples/OpenALTest/OpenALTestBuildScene.o
	${CXX} ${CPPFLAGS} $? ${LDFLAGS} ${LIBS} ${OPENCV_LIBS} ${OPENAL_LIBS} -o $@

Example02: ${COMMONOBJS} ${LIB} ../examples/Example02/Example02.o
	${CXX} ${CPPFLAGS} $? ${LDFLAGS} ${LIBS} -o $@

${LIB}: ${OBJS}
	${AR} rcs ${LIB} $? 

clean:
	@-rm -rf Example*
	@-rm -rf ${COMMONOBJS}
	@-rm -rf ${OBJS}
	@-rm -rf ${LIB}

#xgl.o: xgl.h
#xglshader.o: xgl.h
