#-----------------------------------------------
# Define build settings for basic XClass services
# ie: that which is not dependent on 3rd party
# code that is separate from the MadStyle TV
# project.
#-----------------------------------------------
export

XCLASSBASELIB=libxclass_base.a
XCLASSLIB=libxclass.a

# map environment variables to internal variables
XCLASSDIR=${TOPDIR}xclass
XGLDIR=${XCLASSDIR}/xgl
XALDIR=${XCLASSDIR}/xal
XAVDIR=${XCLASSDIR}/xav
THIRDPARTYDIR=${XCLASSDIR}/3rdParty

CXXFLAGS +=-std=c++14\
	-I${XALDIR} \
	-I${XAVDIR} \
	-I${XGLDIR} \
	-I${XGLDIR}/glm \
	-I${XGLDIR}/glm/gtc \
	-I${THIRDPARTYDIR}/soil/src \
	-I${THIRDPARTYDIR}/freetype/include/freetype2 \
	-I${THIRDPARTYDIR}/mavlink/c_library_v1 \
	-I${THIRDPARTYDIR}/ftdi/include \
	-I${THIRDPARTYDIR}/ftdi/include/linux \
	-I${THIRDPARTYDIR}/mosquitto/lib \
	-I${XCLASSDIR} \
	-I/usr/include/AL \
	-I/usr/local/include/AL \
	-I${THIRDPARTYDIR}/openal-soft/include/AL \
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

LDFLAGS +=\
	-L${XCLASSDIR} \
	-L${THIRDPARTYDIR}/glfw/lib \
	-L${THIRDPARTYDIR}/mosquitto/lib \
	-L/usr/local/lib \
	-L/usr/lib/x86_64-linux-gnu

LIBS =-lxclass -lglfw3 -lGLU -lGL -lX11 -ldl -lXxf86vm -lX11 -lXrandr -lXi -lXinerama -lXcursor -lpthread -lexpat -lfreetype -lmosquitto

#------------------------------------------------------------------------
# if ffmpeg development libraries are installed, uncomment the following
# to enable projects that utilize them. See ReadMe.txt in this directory
#------------------------------------------------------------------------
ifdef HAS_FFMPEG
LIBS +=-lavcodec -lavformat -lavutil
CXXFLAGS += -DHAS_FFMPEG
CFLAGS += -DHAS_FFMPEG
endif
