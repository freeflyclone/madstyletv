/**************************************************************
** NetCDF: demonstrate use of NetCDF big data library
**************************************************************/
#include "ExampleXGL.h"
#include "netcdf.h"

void NetCDFInit(std::string file) {
	int status, ncid, ndims, nvars, ngatts, unlimdimid;

	if ((status = nc_open(file.c_str(), NC_NOWRITE, &ncid)) != NC_NOERR)
	{
		xprintf("%s: nc_open() failed: %d\n", __FUNCTION__, status);
		return;
	}
	
	if ((status = nc_inq(ncid, &ndims, &nvars, &ngatts, &unlimdimid)) != NC_NOERR)
	{
		xprintf("%s: nc_inq() failed: %d\n", __FUNCTION__, status);
		nc_close(ncid);
		return;
	}

	xprintf("%s(): nDims: %d, nvars: %d, ngatts: %d, unlimdimid: %d\n", __FUNCTION__, ndims, nvars, ngatts, unlimdimid);

	nc_close(ncid);
}

void ExampleXGL::BuildScene() {
	XGLShape* shape;

	std::string netCdfPath = config.WideToBytes(config.Find(L"NetCDFDir")->AsString());
	std::string netCdfFile = config.WideToBytes(config.Find(L"NetCDFFile")->AsString());

	NetCDFInit(netCdfPath + netCdfFile);

	AddShape("shaders/000-simple", [&]() { shape = new XGLTriangle(); return shape; });
}