#ifndef XSQLITE_H
#define XSQLITE_H

#include "sqlite3.h"

class Xsqlite
{
public:
	Xsqlite(std::string dbName);
	~Xsqlite();

private:
	std::string dbName;
	sqlite3 *db;
};

#endif // XSQLITE_H