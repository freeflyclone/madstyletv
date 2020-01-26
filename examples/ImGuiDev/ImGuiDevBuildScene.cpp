/**************************************************************
** ImGuiDevBuildScene.cpp
**
** ImGui is a 3rd-party GUI library with tremendous appeal for
** me:  I REALLY don't want to write a GUI layer, because
** writing GUI widgets is way too tedious. ImGui looks like
** it can be made to be pretty enough for professional looking
** UI experiences, which I care about.
**************************************************************/
#include "ExampleXGL.h"
#include "xsqlite.h"
#include "xbento4_class.h"

Xsqlite* xdb;
XBento4* xb4;

void ExampleXGL::BuildScene() {
	std::string dbPath = pathToAssets + "/assets/dbTest.sq3";

	xdb = new Xsqlite(dbPath);

	xdb->AddCallback(
		[&](int argc, char**argv, char** columnNames) 
	{
		Xsqlite::KeyValueList kl;

		for (int i = 0; i < argc; i++)
			kl.push_back({ columnNames[i], argv[i] });

		std::string row;
		for (Xsqlite::KeyValue k : kl)
			row += k.first + ": " + k.second + ", ";

		xprintf("%s\n", row.c_str());

		return 0; 
	});

	std::string  sql = "DROP TABLE IF EXISTS Cars;"
				"CREATE TABLE Cars(Id INT, Name TEXT, Price INT);"
				"INSERT INTO Cars VALUES(1, 'Audi', 52642);"
				"INSERT INTO Cars VALUES(2, 'Mercedes', 57127);"
				"INSERT INTO Cars VALUES(3, 'Skoda', 9000);"
				"INSERT INTO Cars VALUES(4, 'Volvo', 29000);"
				"INSERT INTO Cars VALUES(5, 'Bentley', 350000);"
				"INSERT INTO Cars VALUES(6, 'Citroen', 21000);"
				"INSERT INTO Cars VALUES(7, 'Hummer', 41400);"
				"INSERT INTO Cars VALUES(8, 'Volkswagen', 21600);";

	xdb->Execute(sql);

	xdb->Execute("SELECT name FROM sqlite_master WHERE type = 'table';");
	xdb->Execute("SELECT * FROM Cars;");

	AddShape("shaders/yuv", [&]() { xb4 = new XBento4("H:/Hero6/GH010171.mp4"); return xb4; });
	xb4->Start();
}
