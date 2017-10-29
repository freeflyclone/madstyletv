#ifndef XUTILS_H
#define XUTILS_H

#include <stdlib.h>

//#define printf xprintf

#ifdef __cplusplus

const double PI = 3.141592653589793238463;
const float  PI_F = 3.14159265358979f; 

extern "C" {

#endif

int xprintf(const char *,...);

void sdldie(char *msg);

#ifdef __cplusplus
};
#endif

#endif
