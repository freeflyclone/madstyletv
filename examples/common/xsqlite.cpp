#include "ExampleXGL.h"

#include "xsqlite.h"

Xsqlite::Xsqlite(std::string dbn) : dbName(dbn)
{
	xprintf("%s(\"%s\")\n", __FUNCTION__, dbName.c_str());

	int rc = sqlite3_open(dbName.c_str(), &db);

	if (rc)
	{
		xprintf("sqlite3_open() failed: %d\n", rc);
	}
	else
		xprintf("sqlite3_open() success!\n");
}

Xsqlite::~Xsqlite()
{
	xprintf("%s\n", __FUNCTION__);
}