## MadStyle TV - Open source engine for non-gaming applications
### OpenGL, C++11, multi-threaded, and cross platform
This project provides a cross-platform OpenGL / Multimedia framework, to serve as examples for learning C++, OpenGL and multimedia technologies.

OpenGL 3.2 core profile at a minimum is required, to be compatible with OSX.

C++11 is the version of choice. Prior to C++11, support for C++ threads couldn't be counted on across all target platforms, particularly Windows.  With C++11, std::thread is adequately supported.

Makefiles are provided for Linux building from the command line. The makefiles support the -j option, and the entire project builds in under a minute, including building dependent 3rdParty projects from source. A modern Linux distro with standard development tools installed is assumed, it's been developed on Ubuntu 16.04.  GCC version 5 or greater is required.

Project build files for Visual Studio 2013 are provided, as that version is available for free as the Community Edition.  

XCode project files for OSX users.  I don't claim to be an XCode wiz, there is probably a more elegant structure.

OpenAL, OpenCV and MAVLINK libraries are provided as git submodules.

Short demo videos are available on [YouTube](https://www.youtube.com/user/freeflyclone):
* [XGL Framework Demo](https://www.youtube.com/watch?v=pleL5WhYqtw)
* [XGL : GPU Motion Estimation](https://www.youtube.com/watch?v=bW9WzMeHrvI)
* [XGL : GoPro Footage Demo](https://www.youtube.com/watch?v=XIiSj0IpTiE)
* [XGL : PhysX Integration](https://www.youtube.com/watch?v=FxgMU4fQaCU)
* [XGL : Mavlink](https://www.youtube.com/watch?v=AA7rEu70190)

This repo uses git submodules for some of the larger 3rdParty source code.  I don't much care for git submodules, but haven't figured out a viable alternative.  Anyway here's what you need to know:

  [Cloning with submodules](http://stackoverflow.com/questions/3796927/how-to-git-clone-including-submodules)
