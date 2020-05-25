/**************************************************************
** NetCDF: demonstrate use of NetCDF big data library
**************************************************************/
#include "ExampleXGL.h"
#include "netcdf.h"
#include "xglgraph.h"

namespace {
	// number of days from 1/1/1600 to 1/1/1752
	const int jan1st1752{ 55518 };

	// number of days from 1/1/1600 to 9/2/1752
	const int parliamentOffset{ 55762 };

	class OrdinalToDateConverter {
	public:
		typedef std::map<int, int> Ord2Mon;
		Ord2Mon o2m, o2mLeap;

		OrdinalToDateConverter() {
			int ordinal{ 0 };
			int ordinalLeap{ 0 };

			for (int i = 0; i < 12; i++) {
				for (int j = 0; j < monthLengths[0][i]; j++)
					o2m[ordinal++] = i;
				for (int j = 0; j < monthLengths[1][i]; j++)
					o2mLeap[ordinalLeap++] = i;
			}
		}

		int GetMonth(int ordinal, bool isLeap)
		{
			return isLeap ? o2mLeap[ordinal % 366] : o2m[ordinal % 365];
		}

		int GetMonthDay(int ordinal, bool isLeap)
		{
			return ordinal - monthOffsets[isLeap ? 1 : 0][GetMonth(ordinal, isLeap)];
		}

	private:
		// typed in by hand from ISO 8601 tables
		int monthLengths[2][12] =
		{
			{ 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 },
			{ 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 }
		};
		int monthOffsets[2][12] = {
			{ 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334	},
			{ 0, 31, 60, 91, 121, 152, 183, 213, 244, 274, 305, 335 }
		};
	};

	OrdinalToDateConverter ordToDate;
};

struct GregorianDate {
	int year, month, day, ordinal;

	GregorianDate operator +(int offset) {
		const float yearLengthInDays = 365.2422;
		GregorianDate date{ 0,0,0,0 };

		// September 2nd, 1752: Julian -> Gregorian change mandated
		// by British Parliament takes effect: next day is September 14th, 1752;
		// 11 days are subracted, which is 1 more than the adjustment previously
		// baked into the Gregorian calendar in 1582, when first adopted by Pope
		// Gregory XIII. North America was still mostly a colony of The Empire, so
		// subject to the rule of law from British Parliament.
		if (offset > parliamentOffset)
			offset--;

		float fNumYears = offset / yearLengthInDays;
		float fYearRemainder = fNumYears - floor(fNumYears);
		int numYears = (int)floor(fNumYears);
		int numDays = (int)(fYearRemainder * yearLengthInDays);

		bool isLeapYear = (((numYears % 4) == 0) && !((numYears % 100) == 0)) || ((numYears % 400) == 0);

		date.year = year + numYears;
		date.ordinal = numDays;
		date.month = ordToDate.GetMonth(date.ordinal, isLeapYear) + 1;
		date.day = ordToDate.GetMonthDay(date.ordinal, isLeapYear) + 1;

		return date;
	};

	void print() {
		xprintf("Ordinal day: %d, %02d/%02d/%04d\n", ordinal, day, month, year);
	}
};

struct Epoch {
	GregorianDate d;
	GregorianDate operator +(int o)	{ return d + o;	}
};

Epoch istiEpoch{1600, 1, 1, 0};

struct ISTIObservation {
	GregorianDate date;
	float averageMonthlyTemperature;
};

typedef std::vector<ISTIObservation> ISTIObservations;

class NetCDFData : public ISTIObservations {
public:
	 NetCDFData(std::string file) {
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

		for (int i = 0; i < ndims; i++)
		{
			size_t dimlen = 0;
			if ((status = nc_inq_dimlen(ncid, i, &dimlen)) != NC_NOERR)
			{
				xprintf("%s: nc_inq_unlimdim() failed: %d\n", __FUNCTION__, status);
				return;
			}
			xprintf("%s: dimension[%d]: %d\n", __FUNCTION__, i, dimlen);
		}

		// Question: how do I know "1" is the dimension ID I care about for "variable" length?
		// Answer: Each "variable" can be multidimensional, so each must have its dimensions
		//         precisely specified, and that specification involves *references* to the file's
		//		   global list of possible dimension lengths.  "1" was observed to be the dimension
		//		   ID used by the variables in the NetCDF files I care about, 
		//		   ie: avg temperature time series from ISTI databank.
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

		if (true)
		{
			int surfaceAvgTempId;
			float* avgTemps = new float[nObs];
			int timeId;
			int* times = new int[nObs];

			if ((status = nc_inq_varid(ncid, "time", &timeId)) != NC_NOERR)
			{
				xprintf("%s(): nc_inq_varid() failed: %d\n", __FUNCTION__, status);
			}
			else {
				if ((status = nc_inq_varid(ncid, "surface_average_temperature", &surfaceAvgTempId)) != NC_NOERR)
				{
					xprintf("%s(): nc_inq_varid() failed %d\n", __FUNCTION__, status);
				}
				else
				{
					if ((status = nc_get_var_int(ncid, timeId, times)) != NC_NOERR)
					{
						xprintf("%s(): nc_get_var_int() failed: %d\n", __FUNCTION__, status);
					}
					else
					{
						if ((status = nc_get_var_float(ncid, surfaceAvgTempId, avgTemps)) != NC_NOERR)
						{
							xprintf("%s(): nc_get_var_float() failed: %d\n", __FUNCTION__, status);
						}
						else {
							for (int i = 0; i < nObs; i++)
							{
								GregorianDate d = istiEpoch + times[i];

								push_back({d, avgTemps[i]});
							}
						}
					}
				}
			}
			delete avgTemps;
		}
		nc_close(ncid);
	}

private:
};

XGLGraph* temperatureGraph;
std::vector<float> temps;

void ExampleXGL::BuildScene() {
	XGLShape* shape;
	glm::mat4 translate;

	std::string netCdfPath = config.WideToBytes(config.Find(L"NetCDFDir")->AsString());
	std::string netCdfFile = config.WideToBytes(config.Find(L"NetCDFFile")->AsString());

	NetCDFData nc(netCdfPath + netCdfFile);
	for (auto o : nc)
		temps.push_back(o.averageMonthlyTemperature);

	xprintf("nc has %d entries\n", nc.size());

	AddShape("shaders/000-simple", [&]() { shape = new XGLTriangle(); return shape; });

	AddShape("shaders/000-attributes", [&]() { 
		temperatureGraph = new XGLGraph(temps); 
		return temperatureGraph; 
	});

	temperatureGraph->attributes.diffuseColor = XGLColors::green;
	translate = glm::translate(glm::mat4(), glm::vec3(0.0, 15, 0));
	temperatureGraph->model = translate;
}