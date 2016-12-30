## MadStyle TV - Open Source framework/engine for non-gaming applications
### OpenGL, C++11, multi-threaded, and cross platform
The goal of the project is to provide a cross-platform OpenGL / Multimedia framework, to serve as a guide for learning C++, OpenGL and multimedia technologies.

OpenGL 3.2 core profile at a minimum, is utilized to be compatible with OSX.

C++11 is the version of choice. Prior to C++11, support for C++ threads couldn't be counted on across all target platforms, particularly Windows.  With C++11, std::thread is adequately supported.

Project build files for Visual Studio 2013 are provided, as that version is available for free as the Community Edition.  

XCode project files for OSX users.  I don't claim to be an XCode wiz, there is probably a more elegant structure.

Makefiles are provided for Linux building from the command line. The makefiles support the -j option, and the entire project builds in under a minute, including building dependent 3rdParty projects from source.

OpenAL, OpenCV and MAVLINK libraries are provided as git submodules.  

