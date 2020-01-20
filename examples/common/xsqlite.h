#ifndef XSQLITE_H
#define XSQLITE_H

#include "sqlite3.h"

class Xsqlite
{
public:
	Xsqlite(std::string dbName);
	~Xsqlite();

	int AddCars();

private:
	std::string dbName;
	sqlite3 *db{ nullptr };
};

#endif // XSQLITE_H