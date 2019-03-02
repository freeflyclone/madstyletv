/**************************************************************
** NanoSvgTestBuildScene.cpp
**
** Just to demonstrate instantiation of a "ground"
** plane and a single triangle, with default camera manipulation
** via keyboard and mouse.
**************************************************************/
#include "ExampleXGL.h"

#define NANOSVG_ALL_COLOR_KEYWORDS	// Include full list of color keywords.
#define NANOSVG_IMPLEMENTATION		// Expands implementation
#include "nanosvg.h"
#include "nanosvgrast.h"

int numPoints = 0;
int num2draw;

class NanoSVGShape : public XGLShape {
public:
	class Path {
	public:
		XGLVertex ComputeCentroid(bool *);
		XGLVertexList vl;
	};
	typedef std::vector<Path> PathOutline;

	void Points2XGLVertices(float *p, int npts) {
		//xprintf("npts: %d, %d curves, %d remaining points\n", npts, npts / 4, npts % 4);

		for (auto i = 0; i < npts - 1; i++) {
			auto j = (i + 1) % npts;
			XGLVertex p1 = { -p[i * 2], p[i * 2 + 1], 0 };
			XGLVertex p2 = { -p[j * 2], p[j * 2 + 1], 0 };

			//xprintf("  %0.3f, %03f\n", p1.x, p1.y);

			pathOutline[pathIdx].vl.push_back({ { p1.x, p1.y, 0 } });
			pathOutline[pathIdx].vl.push_back({ { p2.x, p2.y, 0 } });

			numPoints += 2;
		}
	}

	void PathToPoints(float *p, int npts) {
		//xprintf("npts: %d, %d curves, %d remaining points\n", npts, npts / 4, npts % 4);

		for (auto i = 0; i < npts-1; i+=3) {
			auto j = (i + 1) % npts;
			auto k = (j + 1) % npts;
			auto l = (k + 1) % npts;

			XGLVertex p1 = { p[i * 2], -p[i * 2 + 1], 0 };
			XGLVertex c1 = { p[j * 2], -p[j * 2 + 1], 0 };
			XGLVertex c2 = { p[k * 2], -p[k * 2 + 1], 0 };
			XGLVertex p2 = { p[l * 2], -p[l * 2 + 1], 0 };

			//xprintf("  %0.3f, %03f - %0.3f, %03f - %0.3f, %03f - %0.3f, %03f", p1.x, p1.y, c1.x, c1.y, c2.x, c2.y, p2.x, p2.y);

			if (IsEqual(p1, c1) && IsEqual(c1, c2) && IsEqual(c2, p2)) {
				//xprintf(" - degenerate cubic bezier\n");
				pathOutline[pathIdx].vl.push_back({ p2 });
				break;
			}
			else
				//xprintf("\n");

			EvaluateCubicBezier(p1, c1, c2, p2);
		}
	}

	NanoSVGShape(std::string fileName) {
		image = nsvgParseFromFile(fileName.c_str(), "in", 100);
		numPoints = 0;
		pathIdx = 0;
		Reset();
		int numShapes{ 0 };
		for (NSVGshape* shape = image->shapes; shape != NULL; shape = shape->next) {
			int numPaths{ 0 };
			for (NSVGpath* path = shape->paths; path != NULL; path = path->next) {
				if (pathIdx)
					pathOutline.push_back(*(new Path()));

				PathToPoints(path->pts, path->npts);

				auto tmpPath = pathOutline[pathIdx];
				size_t pathLength = tmpPath.vl.size();
				for (int i = 0; i < pathLength - 1; i++) {
					int j = (i + 1) % pathLength;
					v.push_back({ { tmpPath.vl[i].v }, {}, {}, { XGLColors::yellow } });
					v.push_back({ { tmpPath.vl[j].v }, {}, {}, { XGLColors::yellow } });
				}
				pathIdx++;
			}
		}
		num2draw = numPoints = v.size();
	}

	void Draw() {
		glDrawArrays(GL_LINES, 0, GLsizei(num2draw));
		GL_CHECK("glDrawPoints() failed");
	}

	~NanoSVGShape() {
		nsvgDelete(image);
	}

	void Reset() {
		for (auto p : pathOutline)
			p.vl.clear();

		pathOutline.clear();
		pathOutline.push_back(*(new Path()));
		pathIdx = 0;
	}

	bool IsEqual(XGLVertex a, XGLVertex b) {
		return (a.x == b.x) && (a.y == b.y);
	}

	const XGLVertex& Interpolate(const XGLVertex& p1, const XGLVertex& p2, float percent) {
		XGLVertex v;
		float diff;

		diff = p2.x - p1.x;
		v.x = p1.x + (diff * percent);
		diff = p2.y - p1.y;
		v.y = p1.y + (diff * percent);

		return v;
	}

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
			pathOutline[pathIdx].vl.push_back({ out });
		}
		if (IsEqual(p3, firstPoint)) {
			xprintf("Final point of Cubic Bezier is coincident with path start, ignoring\n");
			return;
		}
		pathOutline[pathIdx].vl.push_back({ p3 });
	}

private:
	struct NSVGimage* image;
	const float interpolationFactor{ 0.01 };
	PathOutline pathOutline;
	int pathIdx;
	XGLVertex firstPoint;
};

void ExampleXGL::BuildScene() {
	NanoSVGShape *svgShape;

	// get SvgFile from config
	std::string svgFile = "../" + config.WideToBytes(config.Find(L"SvgFile")->AsString());

	AddShape("shaders/000-simple", [&](){ svgShape = new NanoSVGShape(svgFile); return svgShape; });

	// now hook up a GUI slider to control the number of primitives drawn by NanoSVGShape.Draw() method.
	XGLGuiCanvas *sliders = (XGLGuiCanvas *)(GetGuiManager()->FindObject("HorizontalSliderWindow"));
	if (sliders != nullptr) {
		XGLGuiSlider *hs;
		if ((hs = (XGLGuiSlider *)sliders->FindObject("Horizontal Slider 1")) != nullptr) {
			hs->AddMouseEventListener([hs](float x, float y, int flags) {
				if (hs->HasMouse()) {
					num2draw = (int)(hs->Position()*numPoints);
				}
			});
		}
	}
}
