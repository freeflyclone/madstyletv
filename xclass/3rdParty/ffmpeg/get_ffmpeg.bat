rem This will likely only work in Windows 10, 
rem unless you have curl & tar in your path
rem ------------------------------------------
cd %1
if not exist ffmpeg.zip (
	echo "Downloading ffmpeg.zip from libs.madstyle.tv"
	curl http://libs.madstyle.tv/libffmpeg_4.4.r98605_msvc15_x64.zip --output ffmpeg.zip
)

if not exist lib (
	echo "lib folder not found, expanding ffmpeg.zip"
	tar xf ffmpeg.zip
)
else (
	echo "ffmpeg lib folder found, nothing to do"
)
