rem Count the lines of code I wrote in xclass library:
find . -path ./3rdParty -prune -o -path ./include -prune -o -path ./xgl/GL -prune -o -path ./xgl/glm -prune -o -path ./xgl/GLFW -prune -o \( -name \*.cpp -o -name \*.h \) -exec wc -l {} ; | grep -v ew.h | grep -v JSON | gawk -f sum.awk
