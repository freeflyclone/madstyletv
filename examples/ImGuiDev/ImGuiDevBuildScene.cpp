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

Xsqlite* xdb;

void ExampleXGL::BuildScene() {
	std::string dbPath = pathToAssets + "/assets/dbTest.sq3";

	xdb = new Xsqlite(dbPath);
}
