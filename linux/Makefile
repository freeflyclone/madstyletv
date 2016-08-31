PROGRAM=xgl_test
LIB=libxclass.a
XCLASSDIR=../xclass
XGLDIR=../xclass/xgl
VPATH = ../xclass:../xclass/xgl

CPPFLAGS +=-std=c++14\
	-I../xclass/xgl \
	-I../xclass/xgl/glm \
	-I../xclass/xgl/glm/gtc \
	-I../xclass/3rdParty/soil/src \
	-I../xclass/3rdParty/freetype/include/freetype2 \
	-I../xclass \
	-I../examples/common \
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
	${XGLDIR}/xgllight.cpp \
	${XGLDIR}/xglmaterial.cpp \
	${XGLDIR}/xglobject.cpp \
	${XGLDIR}/xglprojector.cpp \
	${XGLDIR}/xglshader.cpp \
	${XGLDIR}/xglshapes.cpp \
	${XGLDIR}/xgluniform.cpp \
	${XCLASSDIR}/xinput.cpp \
	${XCLASSDIR}/xconfig.cpp \


CSOURCES=${XGLDIR}/glew.c \
	${XCLASSDIR}/xutils.c \
	${XCLASSDIR}/3rdParty/soil/src/image_DXT.c \
	${XCLASSDIR}/3rdParty/soil/src/image_helper.c \
	${XCLASSDIR}/3rdParty/soil/src/SOIL.c \
	${XCLASSDIR}/3rdParty/soil/src/stb_image_aug.c \


PROGRAMSOURCES=main.cpp ../examples/common/ExampleXGL.cpp ../examples/Example01/Example01.cpp

OBJS = $(patsubst %.cpp,%.o,${CPPSOURCES})
OBJS += $(patsubst %.c,%.o,${CSOURCES})

PROGRAMOBJS = $(patsubst %.cpp,%.o,${PROGRAMSOURCES})

LDFLAGS +=\
	-L. \
	-L../xclass/3rdParty/glfw/lib \
	-L/usr/lib/x86_64-linux-gnu

LIBS=-lxclass -lglfw3 -lGLU -lGL -lX11 -ldl -lXxf86vm -lX11 -lXrandr -lXi -lXinerama -lXcursor -lpthread -lexpat -lfreetype

${PROGRAM}: ${PROGRAMOBJS} ${LIB}
	${CXX} ${CPPFLAGS} ${PROGRAMOBJS} ${LDFLAGS} ${LIBS} -o ${PROGRAM}

${LIB}: ${OBJS}
	${AR} rcs ${LIB} $? 

clean:
	@-rm -rf ${OBJS}
	@-rm -rf ${LIB}

#xgl.o: xgl.h
#xglshader.o: xgl.h

%.o:%.cpp
	${CXX} ${CPPFLAGS} -c $*.cpp -o $@
%.o:%.c
	gcc ${CFLAGS} -c $*.c -o $@
