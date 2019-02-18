#ifndef TRIANGULATOR_H
#define TRIANGULATOR_H

// triangle.h wants this
#ifndef REAL
#define REAL double
#endif
extern "C" {
	#define ANSI_DECLARATORS
	#include "triangle.h"
};

extern int numPoints;
extern int num2draw;
extern int numPoints2;
extern int num2draw2;

enum polyParseState {
	GET_POINTS_HEADER,
	GET_POINTS,
	GET_SEGMENTS_HEADER,
	GET_SEGMENTS,
	GET_HOLES_HEADER,
	GET_HOLES,
	GET_REGION_HEADER,
	GET_REGIONS
};

class Triangulator : public triangulateio, public XGLShape {
public:
	void ReadPolyFile(std::string fileName, triangulateio* t) {
		std::ifstream file(fileName);
		std::string line;
		polyParseState ps = GET_POINTS_HEADER;
		int numberPointMarkers = 0;
		int numberSegmentMarkers = 0;
		double x, y;
		int lineNum{ 0 };

		while (std::getline(file, line)) {
			std::stringstream ss(line);
			std::vector<std::string> tokens;
			std::string token;
			int index, p1, p2, p3;

			// we know our input is tokenized with whitespace, so the following works.
			while (ss >> token)
				tokens.push_back(token);

			lineNum++;
			switch (ps) {
			case GET_POINTS_HEADER:
				t->numberofpoints = std::stoi(tokens[0]);
				t->numberofpointattributes = std::stoi(tokens[2]);
				numberPointMarkers = std::stoi(tokens[3]);
				t->pointlist = (REAL *)malloc(t->numberofpoints * 2 * sizeof(REAL));
				t->pointattributelist = (REAL *)malloc(t->numberofpoints * t->numberofpointattributes *	sizeof(REAL));
				ps = GET_POINTS;
				break;

			case GET_POINTS:
				index = std::stoi(tokens[0]) - 1;
				x = std::stod(tokens[1]);
				y = std::stod(tokens[2]);
				t->pointlist[index * 2] = x;
				t->pointlist[index * 2 + 1] = y;

				for (int i = 0; i < t->numberofpointattributes; i++)
					t->pointattributelist[index * t->numberofpointattributes + i] = std::stod(tokens[3 + i]);

				if ((1 + index) == t->numberofpoints) {
					ps = GET_SEGMENTS_HEADER;
				}
				break;

			case GET_SEGMENTS_HEADER:
				t->numberofsegments = std::stoi(tokens[0]);
				numberSegmentMarkers = std::stoi(tokens[1]);
				t->segmentlist = (int*)malloc(t->numberofsegments * 2 * sizeof(int));
				ps = GET_SEGMENTS;
				break;

			case GET_SEGMENTS:
				index = std::stoi(tokens[0]) - 1;
				p1 = std::stoi(tokens[1]);
				p2 = std::stoi(tokens[2]);
				// we're using "start from zero" triangulation, input poly file starts from one,
				// so compensate by subtracting 1 from all input point indexes.
				t->segmentlist[index * 2] = p1 - 1;
				t->segmentlist[index * 2 + 1] = p2 - 1;
				if ((1 + index) == t->numberofsegments)
					ps = GET_HOLES_HEADER;
				break;

			case GET_HOLES_HEADER:
				t->numberofholes = std::stoi(tokens[0]);
				t->holelist = (REAL*)malloc(2 * t->numberofholes * sizeof(REAL));
				ps = GET_HOLES;
				break;

			case GET_HOLES:
				index = std::stoi(tokens[0]) - 1;
				x = std::stod(tokens[1]);
				y = std::stod(tokens[2]);
				t->holelist[index * 2] = x;
				t->holelist[index * 2 + 1] = y;
				if ((1 + index) == t->numberofholes)
					ps = GET_REGION_HEADER;
				break;

			default:
				xprintf("The line is: %s\n", line.c_str());
				break;
			}
		}
	}

	Triangulator() {
		struct triangulateio in{ 0 }, mid{ 0 };


		/* Define input points. */
		ReadPolyFile("../assets/test4.poly", &in);

		/* Triangulate the points.  Switches are chosen to read and write a  */
		/*   PSLG (p), number everything from  */
		/*   zero (z),  */
		/*   neighbor list (n).                                              */

		triangulate("Vzp", &in, &mid, nullptr);

		RenderTriangles(mid);

		drawCount = v.size();
		num2draw2 = drawCount;
		numPoints2 = drawCount;
	}

	void Draw() {
		if (v.size()) {
			glDrawArrays(drawMode, 0, num2draw2);
			GL_CHECK("glDrawArrays() failed");
		}
	}

	void RenderTriangles(triangulateio& in) {
		for (int i = 0; i < in.numberoftriangles; i++) {
			for (int j = 0; j < 3; j++) {
				int idx = in.trianglelist[i * 3 + j];
				// modulo trick: get the next (possibly wrapped) vertex of *this* triangle
				int idxNext = in.trianglelist[(i*3 + ((j + 1) % 3))];

				// first point of the line segment of this triangle's edge
				REAL x = in.pointlist[idx * 2];
				REAL y = in.pointlist[idx * 2 + 1];
				v.push_back({ { x, y, 1 }, {}, {}, { XGLColors::white } });

				// for debugging during dev
				if (drawMode == GL_LINES) {
					REAL x2 = in.pointlist[idxNext * 2];
					REAL y2 = in.pointlist[idxNext * 2 + 1];
					v.push_back({ { x2, y2, 1 }, {}, {}, { XGLColors::white } });
				}
			}
		}
	}

	void RenderSegments(triangulateio& t) {
		for (int i = 0; i < t.numberofsegments; i++) {
			for (int j = 0; j < 2; j++) {
				int idx = t.segmentlist[i * 2 + j];

				REAL x = t.pointlist[idx * 2];
				REAL y = t.pointlist[idx * 2 + 1];
				v.push_back({ { x, y, 1 }, {}, {}, { XGLColors::white } });
			}
		}
	}

	void Dump() {
		xprintf("          Number of points: %d\n", numberofpoints);
		xprintf("Number of point attributes: %d\n", numberofpointattributes);
		for (int i = 0; i < numberofpoints; i++)
			xprintf("Point %d: %0.6f, %0.6f\n", i, pointlist[i * 2], pointlist[i * 2 + 1]);

		xprintf("Number of segments: %d\n", numberofsegments);
		for (int i = 0; i < numberofsegments; i++)
			xprintf("Segment %d: %d,%d\n", i, segmentlist[i * 2], segmentlist[i * 2 + 1]);
	}

	void SetDrawCount(GLsizei count){ drawCount = (count<v.size()) ? count : v.size(); }
private:
	GLuint drawMode = GL_TRIANGLES; // could also be GL_LINES
	GLsizei drawCount;
};


#endif