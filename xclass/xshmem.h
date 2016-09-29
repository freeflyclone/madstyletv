#ifndef XSHMEM_H
#define XSHMEM_H

#include <string>

#ifdef WIN32
#include <Windows.h>
#define DEFAULT_FILE_NAME "C:\\vcam_buffer.dat"
#else
#define DEFAULT_FILE_NAME "vcam_buffer.dat"
#endif

// (virtual memory quantum) at the front for IPC usage
#define PAGE_SIZE 4096

// TBD: this is cheesy. Room for one 1080p video frame plus a "page"
#define FILEMAPPING_SIZE (PAGE_SIZE+(1920*1080*4))

typedef struct {
	unsigned int width;
	unsigned int height;
	unsigned int bytesPerPixel;
	unsigned char reserved[4084];
} MAPPED_HEADER;


class XSharedMem {
public:
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
