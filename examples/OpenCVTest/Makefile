include ../../toplevel.mk
include ../../xclass/xclass.mk
include ../examples.mk

PROGRAM=OpenCVTest

LIBS+=-lopencv_core \
	-lopencv_videoio\
	-lopencv_imgcodecs\
	-lopencv_imgproc

#QtCreator likes an 'all' target
all: ${PROGRAM}

${PROGRAM}: ${COMMONOBJS} ${LIB} ${PROGRAM}BuildScene.o
	${CXX} ${CPPFLAGS} $? ${LDFLAGS} ${LIBS} -o $@

clean:
	@-rm -rf ${PROGRAM}
	@-rm -rf ${COMMONOBJS}

