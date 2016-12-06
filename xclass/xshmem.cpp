#include "xutils.h"
#include "xshmem.h"
#include <string>
#include <sstream>

#ifdef WIN32

XSharedMem::XSharedMem(std::string n) : fileBackingName(n) {
	std::wstringstream ws;
	ws << n.c_str();
	std::wstring wideName = ws.str();

	hFile = CreateFile(wideName.c_str(), 
		(GENERIC_READ | GENERIC_WRITE), 
		(FILE_SHARE_READ | FILE_SHARE_WRITE), 
		NULL, 
		OPEN_ALWAYS, 
		FILE_ATTRIBUTE_NORMAL, 
		NULL);

	if (hFile == INVALID_HANDLE_VALUE)
		xprintf("Failed to open file mapping file C:\\vcam_buffer.dat\n");

	hMapping = CreateFileMapping(hFile, NULL, PAGE_READWRITE, 0, fileMappingSize, NULL);
	if (hMapping == NULL)
		xprintf("Failed to creat file mapping\n");
	else {
		mappedHeader = (unsigned char *)MapViewOfFile(hMapping, FILE_MAP_ALL_ACCESS, 0, 0, fileMappingSize);
		if (mappedHeader == NULL) {
			xprintf("MapViewOfFile() failed\n");
		}
		pHeader = (MAPPED_HEADER *)mappedHeader;
		mappedBuffer = mappedHeader + pageSize;
	}
}
#else
XSharedMem::XSharedMem(std::string n) : fileBackingName(n) {
	mappedHeader = new unsigned char[FILEMAPPING_SIZE];
	mappedBuffer = mappedHeader + PAGE_SIZE;
	pHeader = (MAPPED_HEADER *)mappedHeader;
}
#endif
