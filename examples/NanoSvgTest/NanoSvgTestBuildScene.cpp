/**************************************************************
** NanoSvgTestBuildScene.cpp
**
** Use NanoSVG library to parse SVG files and draw them with
** OpenGL. Is *mostly* correct (near as I can tell) at getting
** the outlines right, but filling outlines isn't done yet.
** There's a LOT that needs doing, but this is at least a
** start!  I'll hook this up to Triangle for filling in the
** next iteration of the code.
**************************************************************/
#include "ExampleXGL.h"

#define NANOSVG_ALL_COLOR_KEYWORDS	// Include full list of color keywords.
#define NANOSVG_IMPLEMENTATION		// Expands implementation
#include "nanosvg.h"
#include "nanosvgrast.h"

size_t numPoints = 0;
size_t num2draw;

// derive from XGLShape an object that will hold the geometry
// for an SVG rendering instance.
class NanoSVGShape : public XGLShape {
public:
	class Path {
	public:
		XGLVertex ComputeCentroid(bool *);
		XGLVertexList vl;
		bool isClosed{ false };
	};
	typedef std::vector<Path*> PathOutline;

	// compare two points in 2D for equality
	bool IsEqual(XGLVertex a, XGLVertex b) {
		return (a.x == b.x) && (a.y == b.y);
	}

	// generate PSLG shapes for input to Triangle
	void PathToPoints(float *p, int npts) {
		for (auto i = 0; i < npts-1; i+=3) {
			auto j = (i + 1) % npts;
			auto k = (j + 1) % npts;
			auto l = (k + 1) % npts;

			XGLVertex p1 = { p[i * 2], -p[i * 2 + 1], 0 };
			XGLVertex c1 = { p[j * 2], -p[j * 2 + 1], 0 };
			XGLVertex c2 = { p[k * 2], -p[k * 2 + 1], 0 };
			XGLVertex p2 = { p[l * 2], -p[l * 2 + 1], 0 };

			// emperically discovered that Inkscape font outlines have a degenerate curve at final point
			// AND it's the same point as the start of the path.  Triangle doesn't like coincident points.
			// (or, *I* haven't figured out how to make it cope with them)
			if (IsEqual(p1, c1) && IsEqual(c1, c2) && IsEqual(c2, p2)) {
				(*pathOutline)[pathIdx]->vl.push_back({ p2 });
				break;
			}

			// even straight lines in paths are expressed as Bezier curves
			// (control points are collinear with endpoints)
			EvaluateCubicBezier(p1, c1, c2, p2);
		}

		// closed (looping) path check
		if (IsEqual((*pathOutline)[pathIdx]->vl.front().v, (*pathOutline)[pathIdx]->vl.back().v)) {
			(*pathOutline)[pathIdx]->isClosed = true;
			xprintf("path is closed\n");
		}
	}

	// make a white "canvas" to display the SVG geometry against.
	void PageBackground(int w, int h) {
		// 2 triangles to make a quad as page background
		// (the NanoSVGShape::Draw() method is coded to treat the first 4 vertices as a GL_TRIANGLE_STRIP)
		v.push_back({ { image->width / 100.0f, 0, -0.01 }, {}, {}, { XGLColors::white } });
		v.push_back({ { 0, 0, -0.01 }, {}, {}, { XGLColors::white } });
		v.push_back({ { image->width / 100.0f, -image->height / 100.0f, -0.01 }, {}, {}, { XGLColors::white } });
		v.push_back({ { 0, -image->height / 100.0f, -0.01 }, {}, {}, { XGLColors::white } });
	}

	XGLVertex ZExtrude(XGLVertex v1, XGLVertex v2) {
		XGLVertex n;

		// this would be useful for extruding...
		n = 0.25f * glm::normalize(glm::cross(v1, v2)) + v1;

		// make vertex input order irrelevant
		n.z = abs(n.z);

		v.push_back({ { v1 }, {}, {}, { XGLColors::blue } });
		v.push_back({ { n }, {}, {}, { XGLColors::blue } });

		return n;
	}

	NanoSVGShape(std::string fileName) {
		numPoints = 0;
		pathIdx = 0;
		int numShapes{ 0 };

		Reset();

		image = nsvgParseFromFile(fileName.c_str(), "in", 100);
		PageBackground(image->width, image->height);

		for (NSVGshape* shape = image->shapes; shape != NULL; shape = shape->next) {
			int numPaths{ 0 };
			auto color = shape->stroke.color;
			XGLColor xColor{ (color & 0xFF), (color & 0xFF00) >> 8, (color & 0xFF0000) >> 16, (color & 0xFF000000) >> 24 };
			for (NSVGpath* path = shape->paths; path != NULL; path = path->next) {
				if (pathIdx)
					pathOutline->push_back(new Path());

				// Convert NanoSVG paths to XGLVertex paths (with Z-axis set to zero)
				// evaluating the Bezier curves with piece-wise interpolation.
				PathToPoints(path->pts, path->npts);

				// Copy the XGLVertex version of the converted paths to this objects
				// rendering XGLVertex set for rendering, specifically so that we
				// can close each outline's loop, otherwise Triangle will misbehave.
				auto tmpPath = (*pathOutline)[pathIdx];
				size_t pathLength = tmpPath->vl.size();
				for (int i = 0; i < pathLength -1; i++) {
					int j = (i + 1) % pathLength;
					int k = (i - 1) % pathLength;

					XGLVertex vPrev = tmpPath->vl[k].v;
					XGLVertex vCurr = tmpPath->vl[i].v;
					XGLVertex vNext = tmpPath->vl[j].v;

					//if (tmpPath->isClosed)
					{
						XGLVertex v1 = vNext - vCurr;
						XGLVertex v2 = vPrev - vCurr;
						XGLVertex b;
						if (i)
							b = 0.25f * glm::normalize(glm::normalize(v1) + glm::normalize(v2));
						else
							b = 0.25f * glm::normalize(XGLVertex(v1.y, -v1.x, 0));

						v.push_back({ { vCurr + b }, {}, {}, { XGLColors::red } });
						v.push_back({ { vCurr - b }, {}, {}, { XGLColors::red } });
					}


					// always add the current point on the path to the rendering list
					v.push_back({ { vCurr }, {}, {}, { xColor } });

					// only add the next point if this is a close path OR we're not at the end of the path.
					if ( tmpPath->isClosed || j != 0 )
						v.push_back({ { vNext }, {}, {}, { xColor } });
				}
				pathIdx++;
			}
			numShapes++;
		}
		// total number of points we need to tell OpenGL about
		numPoints += v.size();

		// initialize the number of points to *actually* draw
		// to "all of them"
		num2draw = numPoints;
		Reset();
	}

	void Draw() {
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
		GL_CHECK("glDrawPoints() failed");

		glDrawArrays(GL_LINES, 4, GLsizei(num2draw - 4));
		GL_CHECK("glDrawArrays() failed");

		glPointSize(8.0f);
		GL_CHECK("glPointSize() failed");

		glDrawArrays(GL_POINTS, 4, GLsizei(num2draw - 4));
		GL_CHECK("glDrawArrays() failed");
	}

	~NanoSVGShape() {
		nsvgDelete(image);
	}

	// Reset our internal data structure
	void Reset() {
		if (pathOutline) {
			for (auto p : *pathOutline) {
				p->vl.clear();
				delete p;
			}

			pathOutline->clear();
			delete pathOutline;
		}
		pathOutline = new PathOutline();
		pathOutline->push_back(new Path());
	}

	// return a point that lies on a line segment some percentage between it's two endpoints
	// (used by Bezier evaluator)
	const XGLVertex Interpolate(const XGLVertex& p1, const XGLVertex& p2, float percent) {
		return { p1.x + ((p2.x - p1.x) * percent), p1.y + ((p2.y - p1.y) * percent), 0.0f };
	}

	// NanoSVG always oututs cubic Bezier paths (2 ctrl points)
	void EvaluateCubicBezier(const XGLVertex& p0, const XGLVertex& p1, const XGLVertex& p2, const XGLVertex& p3) {
		float interpolant;
		XGLVertex i0, i1, i2, m0, m1, out;

		for (interpolant = 0.0f; interpolant < 1.0f; interpolant += interpolationFactor) {
			i0 = Interpolate(p0, p1, interpolant);
			i1 = Interpolate(p1, p2, interpolant);
			i2 = Interpolate(p2, p3, interpolant);

			m0 = Interpolate(i0, i1, interpolant);
			m1 = Interpolate(i1, i2, interpolant);

			out = Interpolate(m0, m1, interpolant);
			(*pathOutline)[pathIdx]->vl.push_back({ out });
		}
		if (IsEqual(p3, firstPoint)) {
			xprintf("Final point of Cubic Bezier is coincident with path start, ignoring\n");
			return;
		}
		(*pathOutline)[pathIdx]->vl.push_back({ p3 });
	}

private:
	struct NSVGimage* image;
	const float interpolationFactor{ 0.25f };
	PathOutline* pathOutline = nullptr;
	int pathIdx{0};
	XGLVertex firstPoint;
};

void ExampleXGL::BuildScene() {
	NanoSVGShape *svgShape;

	// get SvgFile from config
	std::string svgFile = "../" + config.WideToBytes(config.Find(L"SvgFile")->AsString());

	AddShape("shaders/000-simple", [&](){ svgShape = new NanoSVGShape(svgFile); return svgShape; });
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0, 0, 0.1));
	svgShape->model = translate;

	// now hook up a GUI slider to control the number of primitives drawn by NanoSVGShape.Draw() method.
	XGLGuiCanvas *sliders = (XGLGuiCanvas *)(GetGuiManager()->FindObject("HorizontalSliderWindow"));
	if (sliders != nullptr) {
		XGLGuiSlider *hs;
		if ((hs = (XGLGuiSlider *)sliders->FindObject("Draw Count")) != nullptr) {
			hs->AddMouseEventListener([hs](float x, float y, int flags) {
				if (hs->HasMouse())
					num2draw = std::max((int)(hs->Position()*numPoints),4);
			});
		}
	}
}
