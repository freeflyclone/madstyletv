#ifndef XSQLITE_H
#define XSQLITE_H

#include "sqlite3.h"

class Xsqlite
{
public:
	typedef std::function<int(int, char**, char**)> CallbackFn;
	typedef std::pair<std::string, std::string> KeyValue;
	typedef std::vector<KeyValue> KeyValueList;

	Xsqlite(std::string dbName);
	~Xsqlite();

	void AddCallback(CallbackFn);
	int Execute(std::string sql);

private:
	std::string dbName;
	sqlite3 *db{ nullptr };
	static int _callback(void*, int, char**, char**);
	std::vector<CallbackFn> _callbackList;
	int Callback(int, char**, char**);
};

#endif // XSQLITE_H