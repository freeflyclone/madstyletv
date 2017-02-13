include ../topheader.mk
include $(TOPDIR)/xclass/xclass.mk

XCLASS_BASE_CPPSOURCES=\
	${XCLASSDIR}/xassets.cpp \
	${XCLASSDIR}/xinput.cpp \
	${XCLASSDIR}/JSON.cpp \
	${XCLASSDIR}/JSONValue.cpp \
	${XCLASSDIR}/xshmem.cpp \
	${XCLASSDIR}/xobject.cpp \
	${XCLASSDIR}/xuart.cpp \
	${XCLASSDIR}/xuartascii.cpp \
	${XCLASSDIR}/xmavlink.cpp \
	${XCLASSDIR}/xmqtt.cpp

XCLASS_BASE_CSOURCES=\
	${XCLASSDIR}/xutils.c

CPPSOURCES=\
	${XCLASS_BASE_CPPSOURCES} \
	${XGLDIR}/xgl.cpp \
	${XGLDIR}/xglbuffer.cpp \
	${XGLDIR}/xglcamera.cpp \
	${XGLDIR}/xglfont.cpp \
	${XGLDIR}/xglframebuffer.cpp \
	${XGLDIR}/xglgui.cpp \
	${XGLDIR}/xgllight.cpp \
	${XGLDIR}/xglmaterial.cpp \
	${XGLDIR}/xglprojector.cpp \
	${XGLDIR}/xglshader.cpp \
	${XGLDIR}/xglshapes.cpp \
	${XGLDIR}/xgluniform.cpp \
	${XGLDIR}/xglworldcursor.cpp \
	${XALDIR}/xal.cpp \
	${XAVDIR}/xavenc.cpp \
# uncomment these to add multimedia playback within the framework.
#	${XAVDIR}/xav.cpp \
#	${XAVDIR}/xavsrc.cpp \


CSOURCES=${XGLDIR}/glew.c \
	${XCLASS_BASE_CSOURCES} \
	${THIRDPARTYDIR}/soil/src/image_DXT.c \
	${THIRDPARTYDIR}/soil/src/image_helper.c \
	${THIRDPARTYDIR}/soil/src/SOIL.c \
	${THIRDPARTYDIR}/soil/src/stb_image_aug.c \

BASE_OBJS = $(patsubst %.cpp,%.o,${XCLASS_BASE_CPPSOURCES})
BASE_OBJS += $(patsubst %.c,%.o,${XCLASS_BASE_CSOURCES})

LIB_OBJS = $(patsubst %.cpp,%.o,${CPPSOURCES})
LIB_OBJS += $(patsubst %.c,%.o,${CSOURCES})

all: ${XCLASSBASELIB} ${XCLASSLIB}

base: ${XCLASSBASELIB}

${XCLASSLIB}: ${LIB_OBJS}
	${AR} rcs ${XCLASSLIB} $? 

${XCLASSBASELIB}: ${BASE_OBJS}
	${AR} rcs ${XCLASSBASELIB} $? 

clean:
	@-rm -rf ${BASE_OBJS}
	@-rm -rf ${LIB_OBJS}
	@-rm -rf ${XCLASSLIB}
	@-rm -rf ${XCLASSBASELIB}

