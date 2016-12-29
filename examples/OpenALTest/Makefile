include ../../toplevel.mk
include ../../xclass/xclass.mk
include ../examples.mk

PROGRAM=OpenALTest

LIBS+=-lopenal

#QtCreator likes an 'all' target
all: ${PROGRAM}

${PROGRAM}: ${COMMONOBJS} ${LIB} OpenALTestBuildScene.o
	${CXX} ${CPPFLAGS} $? ${LDFLAGS} ${LIBS} -o $@

clean:
	@-rm -rf ${PROGRAM}
	@-rm -rf ${COMMONOBJS}

