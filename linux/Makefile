LIB=libxclass.a
XCLASSDIR=../xclass
XGLDIR=../xclass/xgl
VPATH = ../xclass:../xclass/xgl

CPPFLAGS +=-std=c++11\
	-I../xclass/xgl \
	-I../xclass/xgl/glm \
	-I../xclass/xgl/glm/gtc \
	-I../xclass/3rdParty/soil/src \
	-I../xclass/3rdParty/freetype/include/freetype2 \
	-I../xclass \
	-D_GNU_SOURCE=1 \
	-DGLEW_STATIC \
	-Wno-writable-strings \
	-Wno-deprecated-declarations \
	-g

CFLAGS +=-Iglm \
	-I../xclass/xgl \
	-I../xclass/xgl/glm \
	-I../xclass \
	-D_GNU_SOURCE=1 \
	-DGLEW_STATIC \
	-g

CPPSOURCES=${XGLDIR}/xgl.cpp \
	${XGLDIR}/xglbuffer.cpp \
	${XGLDIR}/xglcamera.cpp \
	${XGLDIR}/xgllight.cpp \
	${XGLDIR}/xglmaterial.cpp \
	${XGLDIR}/xglprojector.cpp \
	${XGLDIR}/xglshader.cpp \
	${XGLDIR}/xglshapes.cpp \
	${XGLDIR}/xgluniform.cpp


CSOURCES=${XGLDIR}/glew.c \
	${XCLASSDIR}/xutils.c

OBJS = $(patsubst %.cpp,%.o,${CPPSOURCES})
OBJS += $(patsubst %.c,%.o,${CSOURCES})

LDFLAGS +=-Wl,-L${LOCALDESTDIR}/lib

all: ${LIB}

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
