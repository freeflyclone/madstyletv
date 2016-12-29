include xclass.mk

ifndef XCLASS_DIR
$(error XCLASS_DIR environment variable must be set.  See readme.txt)
endif

CPPSOURCES=${XGLDIR}/xgl.cpp \
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
	${XCLASSDIR}/xassets.cpp \
	${XCLASSDIR}/xinput.cpp \
	${XCLASSDIR}/JSON.cpp \
	${XCLASSDIR}/JSONValue.cpp \
	${XCLASSDIR}/xshmem.cpp \
	${XCLASSDIR}/xobject.cpp \
	${XCLASSDIR}/xuart.cpp \
	${XAVDIR}/xavenc.cpp \
# uncomment these to add multimedia playback within the framework.
#	${XAVDIR}/xav.cpp \
#	${XAVDIR}/xavsrc.cpp \


CSOURCES=${XGLDIR}/glew.c \
	${XCLASSDIR}/xutils.c \
	${THIRDPARTYDIR}/soil/src/image_DXT.c \
	${THIRDPARTYDIR}/soil/src/image_helper.c \
	${THIRDPARTYDIR}/soil/src/SOIL.c \
	${THIRDPARTYDIR}/soil/src/stb_image_aug.c \


OBJS = $(patsubst %.cpp,%.o,${CPPSOURCES})
OBJS += $(patsubst %.c,%.o,${CSOURCES})

all: ${XCLASSLIB}

${XCLASSLIB}: ${OBJS}
	${AR} rcs ${XCLASSLIB} $? 

clean:
	@-rm -rf ${OBJS}
	@-rm -rf ${XCLASSLIB}

