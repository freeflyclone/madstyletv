PROGRAM=OpenCVWebcam

include ../../toplevel.mk
include $(TOPDIR)/xclass/xclass.mk
include $(TOPDIR)/examples/examples.mk

LIBS+=-lopencv_core \
	-lopencv_videoio\
	-lopencv_imgcodecs\
	-lopencv_imgproc

#QtCreator likes an 'all' target
all: ${PROGRAM}

${PROGRAM}: ${EXAMPLESCOMMONOBJS} ${LIB} ${PROGRAM}BuildScene.o
	$(call PROGRAM_BUILD)

clean:
	@-rm -rf ${PROGRAM}
	@-rm -rf ${EXAMPLESCOMMONOBJS}
