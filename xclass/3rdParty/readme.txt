Some libraries here are git submodules.  
If they are not present, example programs 
that utilize them will not be built.

They are:
	mavlink/c_library_v1
	mosquitto
	openal-soft
	opencv

The Makefile in this directory is setup to 
compile these only if they are present.
Be advised: opencv is big and takes awhile to clone and build.

The following commands will initialize them from GitHub:

	git submodule init mavlink/c_library_v1
	git submodule update
	
	git submodule init mosquitto
	git submodule update
	
	git submodule init openal-soft
	git submodule update
	
	git submodule init opencv
	git submodule update
	
VS2013 project files and TortoiseGit procedures are a work in progress as of 4/23/2017
