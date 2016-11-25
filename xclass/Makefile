# map environment variables to internal variables
XCLASSDIR=${XCLASS_DIR}
PHYSXDIR=${PHYSX_DIR}

XGLDIR=${XCLASSDIR}/xgl
XALDIR=${XCLASSDIR}/xal
XAVDIR=${XCLASSDIR}/xav
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



OBJS = $(patsubst %.cpp,%.o,${CPPSOURCES})
OBJS += $(patsubst %.c,%.o,${CSOURCES})

all: ${LIB}

${LIB}: ${OBJS}
	${AR} rcs ${LIB} $? 

clean:
	@-rm -rf ${OBJS}
	@-rm -rf ${LIB}

