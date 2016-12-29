==============================================
    STATIC LIBRARY : xclass Project Overview
==============================================
Encapsulate cross-platform C++11 nuggets in a thin wrapper.

The makefile build system for linux depends on XCLASS_DIR 
environment variable being set.  It should be set to
point to the directory this readme lives in, with
an absolute path.  For example if this directory is
$(HOME)/src/madstyletv/xclass then

	export XCLASS_DIR=$(HOME)/src/madstyle/xclass

in your .bashrc file would do the trick.

If you've installed ffmpeg development libraries to a well
known location, then 

	export HAS_FFMPEG

in your .bashrc will cause the linux build to include
ffmpeg.  This will allow playing of multimedia streams
within the framework.
