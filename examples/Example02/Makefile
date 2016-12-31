PROGRAM=Example02

include ../../toplevel.mk
include $(TOPDIR)/xclass/xclass.mk
include $(TOPDIR)/examples/examples.mk

#QtCreator likes an 'all' target
all: ${PROGRAM}

${PROGRAM}: ${EXAMPLESCOMMONOBJS} ${LIB} ${PROGRAM}.o
	$(call PROGRAM_BUILD)

clean:
	@-rm -rf ${PROGRAM}
	@-rm -rf ${EXAMPLESCOMMONOBJS}

test:
	@echo ${MAKEFILE_LIST}
	@echo number of entries is MAKEFILE_LIST: $(words $(MAKEFILE_LIST))
	@echo TOPDIR: $(TOPDIR)
	@echo XCLASSDIR: $(XCLASSDIR)
	@echo EXAMPLESDIR: $(EXAMPLESDIR)

