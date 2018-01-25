include ../../../../../../topheader.mk
include $(TOPDIR)/xclass/3rdParty/OculusSDK/LibOVR/Projects/Linux/libovr.mk

CPPSOURCES=\
	${LIBOVRDIR}/Src/OVR_CAPI_Util.cpp \
	${LIBOVRDIR}/Src/OVR_StereoProjection.cpp

CSOURCES=\
	${LIBOVRDIR}/Src/OVR_CAPIShim.c \


LIB_OBJS = $(patsubst %.cpp,%.o,${CPPSOURCES})
LIB_OBJS += $(patsubst %.c,%.o,${CSOURCES})

all: ${LIBOVRLIB}

${LIBOVRLIB}: ${LIB_OBJS}
	${AR} rcs ${LIBOVRLIB} $? 

clean:
	@-rm -rf ${LIB_OBJS}
	@-rm -rf ${LIBOVRLIB}

