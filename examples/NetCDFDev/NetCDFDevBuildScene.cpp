/**************************************************************
** NetCDF: demonstrate use of NetCDF big data library
**************************************************************/
#include "ExampleXGL.h"
#include "netcdf.h"

void NetCDFInit(std::string file) {
	int status, ncid, ndims, nvars, ngatts, unlimdimid;
	size_t nObs;

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

	if ((status = nc_inq_dimlen(ncid, 1, &nObs)) != NC_NOERR)
	{
		xprintf("%s: nc_inq_unlimdim() failed: %d\n", __FUNCTION__, status);
		return;
	}
		
	xprintf("%s(): nDims: %d, nvars: %d, ngatts: %d, unlimdimid: %d, nObs: %d\n", __FUNCTION__, ndims, nvars, ngatts, unlimdimid, nObs);

	for (int i = 0; i < nvars; i++)
	{
		char varName[512];
		int natts, type, ndims, dimIds[32];

		if ((status = nc_inq_var(ncid, i, varName, &type, &ndims, dimIds, &natts)) != NC_NOERR)
		{
			xprintf("%s(): nc_inq_var() failed: %d\n", __FUNCTION__, status);
			break;
		}

		xprintf("%s(): var[%d]: %s, type: %d, ndims: %d", __FUNCTION__, i, varName, type, ndims);
		if (ndims)
			xprintf(" dimIds[0]: %d", dimIds[0]);
		xprintf("\n");
	}


	nc_close(ncid);
}

void ExampleXGL::BuildScene() {
	XGLShape* shape;

	std::string netCdfPath = config.WideToBytes(config.Find(L"NetCDFDir")->AsString());
	std::string netCdfFile = config.WideToBytes(config.Find(L"NetCDFFile")->AsString());

	NetCDFInit(netCdfPath + netCdfFile);

	AddShape("shaders/000-simple", [&]() { shape = new XGLTriangle(); return shape; });
}