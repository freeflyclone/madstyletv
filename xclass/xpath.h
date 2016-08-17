/****************************************************************************
**
** Copyright (C) 2015 Evan Mortimore
** All rights reserved.
**
** definitions of std::string derived class for handling paths and
**  filenames in an OS independent way.
**
** Specifically created so I don't have to include boost just to get the
** already existing functonality boost provides, because I think this
** is easy to replicate (I found this code on my first Google hit),
** and I REALLY do not want to include boost in this project, for the sake
** of keeping it simple for those who come after.
****************************************************************************/
#ifndef XPATH_H
#define XPATH_H

template<class T>
T XBaseName(T const & path, T const & delims = "/\\")
{
    return path.substr(path.find_last_of(delims) + 1);
}

template<class T>
T XPathName(T const & path, T const & delims = "/\\")
{
    typename T::size_type const p(path.find_last_of(delims));
    return p > 0 && p != T::npos ? path.substr(0, p) : path;
}

template<class T>
T XRemoveExtension(T const & filename)
{
    typename T::size_type const p(filename.find_last_of('.'));
    return p > 0 && p != T::npos ? filename.substr(0, p) : filename;
}

#endif