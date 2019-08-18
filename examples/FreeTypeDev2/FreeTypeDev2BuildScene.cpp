/**************************************************************
** FreeTypeDev2BuildScene.cpp
**
** Second Freetype rendering experiment.  First one involved
** using "Triangle": a super old school C library for outline
** triangulation.  Great code, but major overkill for fonts.
**
** Previous attempt was expensive CPU triangulation after CPU
** bezier interpolation, with excessive interpolation to yield
** acceptable visual quality. Not good for interactive updates.
**
** This attempt is focusing on a hybrid approach:
** simple (hopefully) CPU triangulation of TTF font outline points
** with GPU shader for bezier curve interpolation.
**************************************************************/
#include "ExampleXGL.h"
#include <string>
#include "XGLFreeType.h"

#include FT_OUTLINE_H

#ifdef FONT_NAME
#undef FONT_NAME
#endif

#define FONT_NAME "C:/windows/fonts/times.ttf"

// New concept: use additional XGLShape(s) to render visualizations of XGLFreeType structures
// while I'm developing my hybrid shader-based Freetype2 renderer.
class XGLFreetypeProbe : public XGLShape {
public:
	XGLFreetypeProbe(XGL* pxgl) : pXgl(pxgl) {
		xprintf("%s()\n", __FUNCTION__);

		pXgl->AddShape("shaders/specular", [&]() { cube = new XGLCube(); return cube; });
		pXgl->AddShape("shaders/specular", [&]() { cubeX = new XGLCube(); return cubeX; });
		pXgl->AddShape("shaders/specular", [&]() { cubeY = new XGLCube(); return cubeY; });
	}

	void Move(XGLVertex v, FT::BoundingBox bb) {
		glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(0.0375, 0.0375, 0.0375));
		glm::mat4 translate = glm::translate(glm::mat4(), v);
		cube->model = translate * scale;

		translate = glm::translate(glm::mat4(), glm::vec3(bb.lr.x, v.y, 0.0));
		cubeX->model = translate * scale;

		translate = glm::translate(glm::mat4(), glm::vec3(v.x, bb.ul.y, 0.0));
		cubeY->model = translate * scale;
	}

	XGL* pXgl{ nullptr };
	XGLCube *cube{ nullptr }, *cubeX{ nullptr }, *cubeY{ nullptr };
};

class XGLFreeType : public FT::GlyphDecomposer,  public XGLShape {
public:
	typedef std::map<FT_ULong, FT_UInt> CharMap;
	typedef std::vector<GLsizei> ContourEndPoints;

	XGLFreeType(XGL* pXgl) {
		FT_UInt gindex = 0;
		FT_ULong charcode = 0;

		if (FT_Init_FreeType(&ft))
			throwXGLException("Init of FreeType failed");

		if (FT_New_Face(ft, FONT_NAME, 0, &face))
			throwXGLException("FT_New_Face() failed " FONT_NAME);

		if (FT_Select_Charmap(face, ft_encoding_unicode))
			throwXGLException("FT_Select_Charmap(UNICODE) failed.");

		// scale so outline coordinate precision is adequately preserved
		FT_Set_Char_Size(face, ftSize, ftSize, ftResolution, ftResolution);

		for (charcode = FT_Get_First_Char(face, &gindex); gindex; charcode = FT_Get_Next_Char(face, charcode, &gindex))
			charMap.emplace(charcode, gindex);

		// add signal to initial Load() to initialize the materials properties.
		v.push_back({});

		pXgl->AddShape("shaders/specular", [&]() { probe = new XGLFreetypeProbe(pXgl); return probe; });
	}

	void PushVertex(XGLVertexAttributes vrtx, bool isClockwise) {
		vrtx.v.x /= scaleFactor;
		vrtx.v.y /= scaleFactor;
		
		vrtx.v.x += advance.x;
		vrtx.v.y += advance.y;

		vrtx.v.z = 0.01f;
		vrtx.c = XGLColors::yellow;

		// this is part of the hybrid approach: help the shaders 
		// understand the glyphs's outline vs holes: holes get negative alpha
		if (!isClockwise)
			vrtx.c.a *= -1.0f;

		v.push_back(vrtx);
	}

	void AdvanceGlyphPosition() {
		advance.x += face->glyph->advance.x / scaleFactor;
		advance.y += face->glyph->advance.y / scaleFactor;
	}

	static bool SortByXCompare(XGLVertexAttributes a, XGLVertexAttributes b) {
		bool xIsEqual = (a.v.x == b.v.x);
		bool xIsLess = (a.v.x < b.v.x);
		bool yIsLess = (a.v.y < b.v.y);

		if (xIsEqual && yIsLess)
			return true;

		if (xIsLess)
			return true;

		return false;
	}

	void RenderText(std::string textToRender) {
		v.clear();
		contourOffsets.clear();

		if (sorted) {
			sorted->clear();
			delete sorted;
		}

		advance = { 0.0f, 0.0f, 0.0f };

		// for each char in string...
		for (char c : textToRender) {
			FT_Load_Glyph(face, charMap[c], FT_LOAD_FORCE_AUTOHINT | FT_LOAD_TARGET_NORMAL);
			Reset();

			// This process results in a set of FT::Contours (single PSLG outlines) for the glyph
			FT_Outline_Decompose(&face->glyph->outline, (FT_Outline_Funcs*)this, (FT_Outline_Funcs*)this);

			int contourIdx = 0;

			// For each contour of the glyph...
			for (FT::Contour contour : Outline()) {
				// mark the beginning of the contour in this shape's CPU-side XGLVertexList
				contourOffsets.push_back((int)v.size());

				// determine if this contour is clockwise or counter-clockwise
				contour.ComputeCentroid();

				// add each contour vertex to this shape's CPU-side XGLVertexList
				for (XGLVertexAttributes vrtx : contour.v)
					PushVertex(vrtx, contour.isClockwise);
			}
			// mark end offset, so display loop can calculate size
			contourOffsets.push_back((int)v.size());
			AdvanceGlyphPosition();
		}

		// save BoundingBox of first outline of first glyph.
		// Freetype presents contours with outlines (vs holes) first.
		// Multiple non-hole outlines are not handled yet.
		bb = Outline()[0].bb;
		bb.ul.x /= scaleFactor;
		bb.ul.y /= scaleFactor;
		bb.lr.x /= scaleFactor;
		bb.lr.y /= scaleFactor;

		// update this shape's VBO with new geometry from the CPU-side XGLVertexList
		// so it will actually be seen.
		Load(shader, v, idx);

		sorted = new XGLVertexList(v);
		std::sort(sorted->begin(), sorted->end() - 2, SortByXCompare);

		return;
	}

	void Draw() {
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		if (contourOffsets.size()) {
			for (int idx = 0; idx < contourOffsets.size() - 1; idx++) {
				GLuint start = contourOffsets[idx];
				GLuint length = contourOffsets[idx + 1] - start;

				glDrawArrays(GL_LINE_LOOP, start, length);
				GL_CHECK("glDrawArrays() failed");
			}
		}
		glDisable(GL_BLEND);

		// if indicatorIdx has changed...)
		if (indicatorIdx != oldIndicatorIdx) {
			// adjust XGLFreetypeProbe to reflect new indicatorIdx
			probe->Move(v[indicatorIdx].v, bb);
			oldIndicatorIdx = indicatorIdx;
		}
	}

	int indicatorIdx{ 0 };
	int oldIndicatorIdx{ -1 };
	bool showTail = false;
	FT::BoundingBox bb;
	
	XGLFreetypeProbe* probe{ nullptr };
	glm::mat4 probeScale;
	glm::mat4 probeTranslate;

private:
	// These 2 numbers help Freetype math have adequate precision...
	const FT_F26Dot6 ftSize{ 1024 };
	const FT_UInt ftResolution{ 1024 };

	// ...while this is used to scale back to XGL preferred size
	REAL scaleFactor{ 3276.8f };

	FT_Library ft;
	FT_Face face;
	FT_GlyphSlot g;
	CharMap charMap;
	XGLVertex advance;
	std::vector<int>contourOffsets;
	XGLVertexList* sorted{ nullptr };
};

static XGLFreeType *pFt;

void ExampleXGL::BuildScene() {
	XGLImGui *ig = nullptr;

	glm::vec3 cameraPosition(-1, -3, 5);
	glm::vec3 cameraDirection = glm::normalize(cameraPosition*-1.0f);
	glm::vec3 cameraUp = { 0, 0, 1 };
	camera.Set(cameraPosition, cameraDirection, cameraUp);

	AddShape("shaders/000-simple", [&](){ pFt = new XGLFreeType(this); return pFt; });

	if ((ig = (XGLImGui *)(GetGuiManager()->FindObject("TitlerGui"))) != nullptr) {
		ig->AddMenuFunc([&]() {
			if (ImGui::Begin("Titler")) {
				if (ImGui::CollapsingHeader("Tweeks", ImGuiTreeNodeFlags_DefaultOpen)) {
					ImGui::SliderInt("Indicator Index", &pFt->indicatorIdx, 0, pFt->v.size()-1);
				}
				ImGui::End();
			}
			else
				ImGui::End();

			return;
		});
	}

	pFt->RenderText("&");
}
