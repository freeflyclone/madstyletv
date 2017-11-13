#-----------------------------------------------
# Define build settings for basic LibOVR 
#-----------------------------------------------
export

LIBOVRLIB=liblibovr.a

# map environment variables to internal variables
XCLASSDIR=${TOPDIR}xclass
THIRDPARTYDIR=${XCLASSDIR}/3rdParty
LIBOVRDIR=${THIRDPARTYDIR}/OculusSDK/LibOVR

CXXFLAGS +=-std=c++14\
	-I${LIBOVRDIR}/Include \
	-DNDEBUG \
	-D_GNU_SOURCE=1 \
	-DGLEW_STATIC \
	-Wno-deprecated-declarations \
	-g

CFLAGS +=-std=c11 \
	-I${LIBOVRDIR}/Include \
	-D_GNU_SOURCE=1 \
	-DGLEW_STATIC \
	-g

LDFLAGS +=\
	-L${LIBOVRDIR}/Projects/Linux \
	-L/usr/local/lib \
	-L/usr/lib/x86_64-linux-gnu

LIBS =-llibovr

#------------------------------------------------------------------------
# if ffmpeg development libraries are installed, uncomment the following
# to enable projects that utilize them. See ReadMe.txt in this directory
#------------------------------------------------------------------------
ifdef HAS_FFMPEG
LIBS +=-lavcodec -lavformat -lavutil
CXXFLAGS += -DHAS_FFMPEG
CFLAGS += -DHAS_FFMPEG
endif
