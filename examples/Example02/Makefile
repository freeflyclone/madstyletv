include ../../toplevel.mk
include ../../xclass/xclass.mk
include ../examples.mk

PROGRAM=Example02

#QtCreator likes an 'all' target
all: ${PROGRAM}

${PROGRAM}: ${COMMONOBJS} ${LIB} Example02.o
	${CXX} ${CPPFLAGS} $? ${LDFLAGS} ${LIBS} -o $@

clean:
	@-rm -rf ${PROGRAM}
	@-rm -rf ${COMMONOBJS}
