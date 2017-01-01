ifeq ($(PHYSX_DIR),)
$(error If you want to build the PhysX example, install the PhysX SDK)
endif

PROGRAM=PhysXTest

include ../../topheader.mk
include $(TOPDIR)/xclass/xclass.mk
include $(TOPDIR)/examples/examples.mk

PHYSXDIR=${PHYSX_DIR}

CXXFLAGS+=-I${PHYSX_DIR}/Include\
	-I.

LDFLAGS+=-L${PHYSX_DIR}/Lib/linux64\
	-L${PHYSX_DIR}/Bin/linux64\


PROGRAM_SOURCES = $(PROGRAM_SOURCES_NO_MAIN) \
	$(PROGRAM)BuildScene.cpp \
	physx-main.cpp \
	physx-xgl.cpp \

LIBS+=-lPhysX3Extensions \
	-lPhysX3Common_x64 \
	-lPhysX3_x64 \
	-lPhysX3Cooking_x64 \
	-lPhysX3CharacterKinematic_x64 \
	-lPhysX3Vehicle \
	-lPhysXProfileSDK \
	-lPhysXVisualDebuggerSDK \
	-lPxTask

include $(TOPDIR)topfooter.mk
