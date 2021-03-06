# build various 3rdParty projects directly without 
# altering them in any way. The default behavior 
# of the build is to copy 3rdParty stuff to /usr/local/...
# which is fairly standard for Linux
# --------------------------------------------------------
include ../../topheader.mk

#set some variables to see if some 3rdParty library source was fetched
OPENAL=$(shell ls openal-soft 2>/dev/null)
OPENCV=$(shell ls opencv 2>/dev/null)
MOSQUITTO=$(shell ls mosquitto 2>/dev/null)
MAVLINK=$(shell ls mavlink/c_library_v1 2>/dev/null)
OPENCV=$(shell ls opencv 2>/dev/null)
LIBOVR=$(shell ls OculusSDK 2>/dev/null)

all: mavlink mosquitto openal opencv libovr

clean:
	@echo making 'clean' in 3rdParty
	@rm -rf openal-soft/build/*
	@-(cd mosquitto && make clean)
	@-(cd OculusSDK/LibOVR/Projects/Linux && make clean)
	#@(if [ -d opencv/build ] ; then cd opencv/build ; make clean ; fi)

.PHONY: mavlink
mavlink:
ifeq ($(MAVLINK),)
	@echo mavlink not found, updating git submodule...
	git submodule init mavlink ; git submodule update mavlink
	@echo ..update done.
endif

.PHONY: mosquitto
mosquitto:
ifeq ($(MOSQUITTO),)
	@echo mosquitto folder not found, updating git submodule...
	git submodule init mosquitto ; git submodule update mosquitto
endif
	(cd mosquitto && make WITH_SRV=no WITH_UUID=no binary && sudo make WITH_DOCS=no install)

openal:
ifeq ($(OPENAL),)
	@echo openal-soft not found, updating git submodule...
	git submodule init openal-soft ; git submodule update openal-soft
endif
	(cd openal-soft/build ; cmake .. ; make -j12 ; sudo make install) 

.PHONY: opencv
opencv:
ifeq ($(OPENCV),)
	@echo opencv not found, updating git submodule...
	git submodule init opencv ; git submodule update opencv
endif
	# OpenCV is a bit of a monster, running cmake invokes 
	# a lengthy download, so avoid it if it's not necessary
	@(if [ ! -d opencv/build ] ;\
	then echo opencv/build NOT found ;\
	 	mkdir opencv/build ; \
	 	cd opencv/build ; \
	 	cmake .. ; \
		make -j 12; \
		sudo make install ; \
	else \
		cd opencv/build ; \
		make -j 12; \
		sudo make install ; \
	fi)

.PHONY: libovr
libovr:
	(cd OculusSDK/LibOVR/Projects/Linux ; make -j12)

