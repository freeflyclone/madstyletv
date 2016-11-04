rem count lines of code I've written:
find . \( -name \*.cpp -o -name \*.h \) -exec wc -l {} ; | gawk -f ../xclass/sum.awk
