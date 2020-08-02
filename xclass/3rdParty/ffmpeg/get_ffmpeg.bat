rem This will likely only work in Windows 10, 
rem unless you have curl & tar in your path
rem ------------------------------------------
curl http://libs.madstyle.tv/libffmpeg_4.4.r98605_msvc15_x64.zip --output ffmpeg.zip
tar xf ffmpeg.zip
del ffmpeg.zip
