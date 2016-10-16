/****************************************************************************
**
** Copyright (C) 2010 Evan Mortimore
** All rights reserved.
**
****************************************************************************/
#ifndef XGLEXCEPT_H
#define XGLEXCEPT_H

#include <stdexcept>

void CheckGlError(const char *, int, std::string);
void CheckGlStatus(const char *, int);

#ifdef _DEBUG
#define GL_CHECK(what) CheckGlError(__FILE__,__LINE__,what)
#define GL_STATUS() CheckGlStatus(__FILE__,__LINE__);
#else
#define GL_CHECK(what)
#define GL_STATUS()
#endif

class XGLException: public std::runtime_error
{
	std::string msg;

public:
	XGLException(const char *file, const int line, const std::string &arg="") : 
		runtime_error(arg)
	{
		std::ostringstream o;
		o << std::string("XGLException: ")  << "@ " << file << ":" << line << "\n\n" << arg;
		msg = o.str();
	}
    ~XGLException() throw() {};

	const char *what() const throw () {
		return msg.c_str();
	}
};

#define throwXGLException(x) throw XGLException(__FILE__, __LINE__, x)

#endif // XGLEXCEPT_H
