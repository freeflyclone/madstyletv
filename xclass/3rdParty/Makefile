# build various 3rdParty projects directly without 
# altering them in any way. The default behavior 
# of the build is to copy 3rdParty stuff to /usr/local/...
# which is fairly standard for Linux
# --------------------------------------------------------
include ../../topheader.mk

OPENAL=$(shell ls openal-soft)
OPENCV=$(shell ls opencv)

all:
ifeq ($(OPENCV),)
	@echo opencv is not present.
else
	@echo opencv is present
	# OpenCV is a bit of a monster, running cmake invokes 
	# a lengthy download, so avoid it if it's not necessary
	@(if [ ! -d opencv/build ] ;\
	then echo opencv/build NOT found ;\
	 	mkdir opencv/build ; \
	 	cd opencv/build ; \
	 	cmake .. ; \
	else \
		cd opencv/build ; \
		make -j 12; \
		sudo make install ; \
	fi)
endif
ifeq ($(OPENAL),)
	@echo openal-soft is not present.
else
	(cd openal-soft/build ; cmake .. ; make ; sudo make install) 
endif


clean:
	@echo making 'clean' in 3rdParty
	@rm -rf openal-soft/build/*
	@(if [ -d opencv/build ] ; then cd opencv/build ; make clean ; fi)

