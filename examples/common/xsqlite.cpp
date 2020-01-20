#include "ExampleXGL.h"
#include "xsqlite.h"

Xsqlite::Xsqlite(std::string dbn) : dbName(dbn)
{
	int rc = sqlite3_open(dbName.c_str(), &db);

	if (rc)
		throw std::runtime_error("sqlite3_open() failed");
}

Xsqlite::~Xsqlite()
{
	if (db)
		sqlite3_close(db);
}

int Xsqlite::Execute(std::string sql) {
	char* errorMsg{ nullptr };

	if (!db)
		throw std::runtime_error("Xsqlite::db is not open");

	int rc = sqlite3_exec(db, sql.c_str(), _callback, this, &errorMsg);
	if (rc != SQLITE_OK) {
		xprintf("%s(): error: %s\n", __FUNCTION__, errorMsg);
		sqlite3_free(errorMsg);
	}

	return rc;
}

void Xsqlite::AddCallback(CallbackFn fn)
{
	_callbackList.push_back(fn);
}

int Xsqlite::Callback(int argc, char** argv, char** columnNames)
{
	int rc = 0;

	for (CallbackFn cb : _callbackList)
		rc = cb(argc, argv, columnNames);

	return rc;
}

int Xsqlite::_callback(void* pCtx, int argc, char** argv, char** columnNames)
{
	Xsqlite* pSqlite = (Xsqlite*)pCtx;

	if (pCtx && pSqlite->_callbackList.size())
		return ((Xsqlite*)(pCtx))->Callback(argc, argv, columnNames);

	return SQLITE_OK;
}
