#ifndef XSHMEM_H
#define XSHMEM_H

#include <string>

#ifdef WIN32
#include <Windows.h>
	const std::string shmemDefaultFile("C:\\vcam_buffer.dat");
#else
const std::string shmemDefaultFile("vcam_buffer.dat");
#endif

class XSharedMem {
public:
	static const int pageSize = 4096;
	static const int fileMappingSize = pageSize + 1920 * 1080 * 4;
	typedef struct {
		unsigned int width;
		unsigned int height;
		unsigned int bytesPerPixel;
		unsigned char reserved[pageSize - 3*sizeof(unsigned int)];
	} MAPPED_HEADER;

	XSharedMem(std::string n);
	
	unsigned char *mappedHeader;
	unsigned char *mappedBuffer;
	MAPPED_HEADER *pHeader;

private:
	std::string fileBackingName;
#ifdef WIN32
	HANDLE hFile, hMapping;
#else
	int hFile, hMapping;
#endif
};

#endif
