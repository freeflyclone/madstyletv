#include "Triangulator.h"

void Triangulator::Free(triangulateio& t, bool flag) {
	if (t.pointlist) free(t.pointlist);
	if (t.pointattributelist) free(t.pointattributelist);
	if (t.pointmarkerlist) free(t.pointmarkerlist);
	if (t.trianglelist)	free(t.trianglelist);
	if (t.triangleattributelist) free(t.triangleattributelist);
	if (t.trianglearealist) free(t.trianglearealist);
	if (t.neighborlist) free(t.neighborlist);
	if (t.segmentlist) free(t.segmentlist);
	if (t.segmentmarkerlist) free(t.segmentmarkerlist);
	if (t.edgelist) free(t.edgelist);
	if (t.edgemarkerlist) free(t.edgemarkerlist);
	if (t.normlist) free(t.normlist);

	// so-called "out" triangulateio struct have some ptrs copied from "in", 
	// avoid double free() if caller (who knows which is which) says to. 
	if (flag) {
		if (t.holelist) free(t.holelist);
		if (t.regionlist) free(t.regionlist);
	}

	Init(t);
}

void Triangulator::Init(triangulateio& t) {
	memset(&t, 0, sizeof(t));
}

void Triangulator::Convert(FT::GlyphOutline& ftGlyphOutline, XGLVertex& advance) {
	Init(*this);

	size_t numPoints = 0, numSegments;

	for (FT::Contour contour : ftGlyphOutline)
		numPoints += contour.v.size();

	numSegments = numPoints;

	pointlist = (REAL*)malloc(2 * sizeof(REAL) * numPoints);
	segmentlist = (int*)malloc(2 * sizeof(int) * numSegments);
	holelist = (REAL*)malloc(2 * sizeof(REAL) * 20);

	int contourOffset = 0;
	int pIdx = 0;
	int sIdx = 0;
	for (FT::Contour contour : ftGlyphOutline)
	{
		size_t numPoints = contour.v.size();
		size_t numSegments = numPoints;

		for (int i = 0; i < numPoints; i++) {
			pointlist[pIdx++] = contour.v[i].v.x / scaleFactor + advance.x;
			pointlist[pIdx++] = contour.v[i].v.y / scaleFactor + advance.y;
		}

		for (int i = 0; i < numPoints; i++) {
			int j = (i + 1) % numPoints;
			segmentlist[sIdx++] = i + contourOffset;
			segmentlist[sIdx++] = j + contourOffset;
		}
		numberofpoints += (int)numPoints;
		numberofsegments += (int)numSegments;

		XGLVertex centroid = contour.ComputeCentroid();

		// if this FT::Contour is CCW, it's a hole to be extracted
		if (!contour.isClockwise) {
			holelist[numberofholes * 2] = centroid.x / scaleFactor + advance.x;
			holelist[numberofholes * 2 + 1] = centroid.y / scaleFactor + advance.y;
			numberofholes++;
		}
		contourOffset += (int)numPoints;
	}

	triangulateio out;
	Init(out);

	triangulate("zpYY", (triangulateio*)this, &out, NULL);

	RenderTriangles(out);

	Free(out, false);
	Free(*this, true);
}

Triangulator::Triangulator() {}

void Triangulator::RenderTriangles(triangulateio& t) {
	for (int i = 0; i < t.numberoftriangles; i++) {
		for (int j = 0; j < 3; j++) {
			int idx = t.trianglelist[i * 3 + j];
			// modulo trick: get the next (possibly wrapped) vertex of *this* triangle
			int idxNext = t.trianglelist[(i * 3 + ((j + 1) % 3))];

			// first point of the line segment of this triangle's edge
			REAL x = t.pointlist[idx * 2];
			REAL y = t.pointlist[idx * 2 + 1];
			v.push_back({ { x, y, 1 }, texCoord, {}, neonYellow });
		}
	}
}
