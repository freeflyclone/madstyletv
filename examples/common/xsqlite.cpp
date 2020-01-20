#include "ExampleXGL.h"

#include "xsqlite.h"

Xsqlite::Xsqlite(std::string dbn) : dbName(dbn)
{
	xprintf("%s(\"%s\")\n", __FUNCTION__, dbName.c_str());

	int rc = sqlite3_open(dbName.c_str(), &db);

	if (rc)
		throw std::runtime_error("sqlite3_open() failed");
}

Xsqlite::~Xsqlite()
{
	xprintf("%s\n", __FUNCTION__);
	if (db) 
	{
		sqlite3_close(db);
	}
}

int Xsqlite::AddCars() {
	char* errorMsg{ nullptr };

	if (!db)
		throw std::runtime_error("Xsqlite::db is not open");

	char *sql = "DROP TABLE IF EXISTS Cars;"
		"CREATE TABLE Cars(Id INT, Name TEXT, Price INT);"
		"INSERT INTO Cars VALUES(1, 'Audi', 52642);"
		"INSERT INTO Cars VALUES(2, 'Mercedes', 57127);"
		"INSERT INTO Cars VALUES(3, 'Skoda', 9000);"
		"INSERT INTO Cars VALUES(4, 'Volvo', 29000);"
		"INSERT INTO Cars VALUES(5, 'Bentley', 350000);"
		"INSERT INTO Cars VALUES(6, 'Citroen', 21000);"
		"INSERT INTO Cars VALUES(7, 'Hummer', 41400);"
		"INSERT INTO Cars VALUES(8, 'Volkswagen', 21600);";

	int rc = sqlite3_exec(db, sql, 0, 0, &errorMsg);

	if (rc != SQLITE_OK) {
		xprintf("%s(): error: %s\n", errorMsg);
		sqlite3_free(errorMsg);
	}
}