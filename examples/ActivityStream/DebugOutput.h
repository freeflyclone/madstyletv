#ifndef DEBUGOUTPUT_H
#define DEBUGOUTPUT_H

#include <ostream>

#include "ExampleXGL.h"

/// \brief This class is derived from basic_stringbuf which will output
/// all the written data using the OutputDebugString function
class OutputDebugStringBuf : public std::basic_stringbuf<char, std::char_traits<char>> {
public:
	explicit OutputDebugStringBuf() : _buffer(8192) 
	{
		setg(nullptr, nullptr, nullptr);
		setp(_buffer.data(), _buffer.data(), _buffer.data() + _buffer.size());
	}

	~OutputDebugStringBuf() {
	}

	static_assert(std::is_same<char, char>::value, "OutputDebugStringBuf only supports char types");

	int sync() {
		try
		{
			MessageOutputer<char, std::char_traits<char>>()(pbase(), pptr());
			setp(_buffer.data(), _buffer.data(), _buffer.data() + _buffer.size());
			return 0;
		}
		catch (...)
		{
			return -1;
		}
	}

	int_type overflow(int_type c = std::char_traits<char>::eof()) 
	{
		auto syncRet = sync();
		if (c != std::char_traits<char>::eof()) {
			_buffer[0] = c;
			setp(_buffer.data(), _buffer.data() + 1, _buffer.data() + _buffer.size());
		}
		return syncRet == -1 ? std::char_traits<char>::eof() : 0;
	}


private:
	std::vector<char> _buffer;

	template<typename TChar, typename TTraits>
	struct MessageOutputer;

	template<>
	struct MessageOutputer<char, std::char_traits<char>> {
		template<typename TIterator>
		void operator()(TIterator begin, TIterator end) const {
			std::string s(begin, end);
			OutputDebugStringA(s.c_str());
		}
	};

	template<>
	struct MessageOutputer<wchar_t, std::char_traits<wchar_t>> {
		template<typename TIterator>
		void operator()(TIterator begin, TIterator end) const {
			std::wstring s(begin, end);
			OutputDebugStringW(s.c_str());
		}
	};
};

extern void InitStdLog();

#endif