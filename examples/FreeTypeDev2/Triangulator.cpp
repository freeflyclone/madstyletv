#include "Triangulator.h"

void Triangulator::Free(triangulateio& t, bool flag) {
	if (t.pointlist)
		free(t.pointlist);

	if (t.pointattributelist)
		free(t.pointattributelist);

	if (t.pointmarkerlist)
		free(t.pointmarkerlist);

	if (t.trianglelist)
		free(t.trianglelist);

	if (t.triangleattributelist)
		free(t.triangleattributelist);

	if (t.trianglearealist)
		free(t.trianglearealist);

	if (t.neighborlist)
		free(t.neighborlist);

	if (t.segmentlist)
		free(t.segmentlist);

	if (t.segmentmarkerlist)
		free(t.segmentmarkerlist);
	
	if (t.holelist && flag)
		free(t.holelist);

	if (t.regionlist)
		free(t.regionlist);

	if (t.edgelist)
		free(t.edgelist);

	if (t.edgemarkerlist)
		free(t.edgemarkerlist);

	if (t.normlist)
		free(t.normlist);

	Init(t);
}
void Triangulator::Init(triangulateio& t) {
	t.pointlist = 0;
	t.pointattributelist = 0;
	t.pointmarkerlist = 0;
	t.numberofpoints = 0;
	t.numberofpointattributes = 0;
	t.trianglelist = 0;
	t.triangleattributelist = 0;
	t.trianglearealist = 0;
	t.neighborlist = 0;
	t.numberoftriangles = 0;
	t.numberofcorners = 0;
	t.numberoftriangleattributes = 0;
	t.segmentlist = 0;
	t.segmentmarkerlist = 0;
	t.numberofsegments = 0;
	t.holelist = 0;
	t.numberofholes = 0;
	t.regionlist = 0;
	t.numberofregions = 0;
	t.edgelist = 0;
	t.edgemarkerlist = 0;
	t.normlist = 0;
	t.numberofedges = 0;
}

void Triangulator::Convert(FT::GlyphOutline& go, triangulateio&  in, XGLVertex& a) {
	Init(in);

	size_t numPoints = 0, numSegments;

	for (auto c : go)
		numPoints += c.v.size();

	numSegments = numPoints;

	in.pointlist = (REAL*)malloc(2 * sizeof(REAL) * numPoints);
	in.segmentlist = (int*)malloc(2 * sizeof(int) * numSegments);
	in.holelist = (REAL*)malloc(2 * sizeof(REAL) * 20);

	int contourOffset = 0;
	int pIdx = 0;
	int sIdx = 0;
	for (FT::Contour c : go)
	{
		size_t numPoints = c.v.size();
		size_t numSegments = numPoints;

		for (int i = 0; i < numPoints; i++) {
			in.pointlist[pIdx++] = c.v[i].v.x / scaleFactor + a.x;
			in.pointlist[pIdx++] = c.v[i].v.y / scaleFactor + a.y;
		}

		for (int i = 0; i < numPoints; i++) {
			int j = (i + 1) % numPoints;
			in.segmentlist[sIdx++] = i + contourOffset;
			in.segmentlist[sIdx++] = j + contourOffset;
		}
		in.numberofpoints += (int)numPoints;
		in.numberofsegments += (int)numSegments;

		bool isClockwise;
		XGLVertex v = c.ComputeCentroid();
		if (!c.isClockwise) {
			in.holelist[in.numberofholes * 2] = v.x / scaleFactor + a.x;
			in.holelist[in.numberofholes * 2 + 1] = v.y / scaleFactor + a.y;
			in.numberofholes++;
		}
		contourOffset += (int)numPoints;
	}
}

Triangulator::Triangulator() {
}

void Triangulator::Draw() {
	if (v.size()) {
		glDrawArrays(drawMode, 0, v.size());
		GL_CHECK("glDrawArrays() failed");
	}
}

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

void Triangulator::RenderSegments(triangulateio& t) {
	for (int i = 0; i < t.numberofsegments; i++) {
		for (int j = 0; j < 2; j++) {
			int idx = t.segmentlist[i * 2 + j];

			REAL x = t.pointlist[idx * 2];
			REAL y = t.pointlist[idx * 2 + 1];
			v.push_back({ { x, y, 1 }, {}, {}, { XGLColors::white } });
		}
	}
}

void Triangulator::Dump(triangulateio& t) {
	xprintf("%d 2 %d 0\n", t.numberofpoints, t.numberofpointattributes);
	for (int i = 0; i < t.numberofpoints; i++)
		xprintf("%d %0.6f %0.6f\n", i + 1, t.pointlist[i * 2], t.pointlist[i * 2 + 1]);

	xprintf("%d 0\n", t.numberofsegments);

	for (int i = 0; i < t.numberofsegments; i++)
		xprintf("%d %d %d\n", i + 1, t.segmentlist[i * 2] + 1, t.segmentlist[i * 2 + 1] + 1);
}

void Triangulator::SetDrawCount(GLsizei count) { 
	drawCount = (count<v.size()) ? count : (GLsizei)v.size(); 
}
