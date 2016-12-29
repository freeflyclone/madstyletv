include ../../xclass/xclass.mk
include ../examples.mk

PROGRAM=MavlinkDev

#QtCreator likes an 'all' target
all: ${PROGRAM}

${PROGRAM}: ${COMMONOBJS} ${LIB} MavlinkDevBuildScene.o xmavlink.o
	${CXX} ${CPPFLAGS} $? ${LDFLAGS} ${LIBS} -o $@

clean:
	@-rm -rf ${PROGRAM}
	@-rm -rf ${COMMONOBJS}

