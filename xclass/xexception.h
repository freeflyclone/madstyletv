#ifndef XEXCEPTION_H
#define XEXCEPTION_H

#include <stdexcept>
#include <sstream>

class XException : public std::runtime_error
{
	std::string msg;

public:
	XException(const char *fileName, const int lineNum, const std::string &arg = "") :
		runtime_error(arg)
	{
		std::ostringstream o;
		o << std::string("XException: ") << "@ " << fileName << ":" << lineNum << "\n\n" << arg;
		msg = o.str();
	}
	~XException() throw() {};

	const char *what() const throw () {
		return msg.c_str();
	}
};

#define throwXException(x) throw XException(__FILE__, __LINE__, x)

#endif // XEXCEPTION
