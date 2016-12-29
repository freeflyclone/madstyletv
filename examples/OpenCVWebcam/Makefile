include ../../toplevel.mk
include ../../xclass/xclass.mk
include ../examples.mk

PROGRAM=OpenCVWebcam

LIBS+=-lopencv_core \
	-lopencv_videoio

#QtCreator likes an 'all' target
all: ${PROGRAM}

${PROGRAM}: ${COMMONOBJS} ${LIB} ${PROGRAM}BuildScene.o
	${CXX} ${CPPFLAGS} $? ${LDFLAGS} ${LIBS} -o $@

clean:
	@-rm -rf ${PROGRAM}
	@-rm -rf ${COMMONOBJS}

