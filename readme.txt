MadStyleTV - Open source project
--------------------------------
The goal of the project is to provide a cross-platform
OpenGL / Multimedia framework, to serve as a guide for
learning C++, OpenGL and multimedia technologies.

OpenGL 3.2 core profile is utilized to be compatible
with OSX.

C++11 is the version of choice. Prior to C++11, support
for C++ threads couldn't be counted on across all target
platforms, particularly Windows.  With C++11, std::thread
is adequately supported.

Project build files for Visual Studio 2013 are provided
as that version is available for free as the Community
Edition.  Also provided are XCode project files for OSX
users.

Makefiles are provided for Linux building from the
command line. QtCreator project files are also provided,
for those who prefer an IDE over command lines.  The
makefiles support the -j option, and the entire tree
builds in seconds on a multi-core machine.

This repository has been migrated from the original
SVN repository.
