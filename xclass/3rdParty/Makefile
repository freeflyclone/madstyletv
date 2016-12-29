# build various 3rdParty projects directly without 
# altering them in any way. The default behavior 
# of the build is to copy 3rdParty stuff to /usr/local/...
# which is fairly standard for Linux
# --------------------------------------------------------
include ../../toplevel.mk
include ../xclass.mk


all: 
	@echo making 'all' in 3rdParty
	(cd openal-soft/build ; cmake .. ; make ; sudo make install)

clean:
	@echo making 'clean' in 3rdParty
	@rm -rf openal-soft/build/*

