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

class XGLFreeType : public FT::GlyphDecomposer,  public XGLShape {
public:
	typedef std::map<FT_ULong, FT_UInt> CharMap;
	typedef std::vector<GLsizei> ContourEndPoints;

	XGLFreeType() {
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

	void RenderText(std::string textToRender) {
		v.clear();
		contourOffsets.clear();

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

		bb = Outline()[0].bb;
		bb.ul.x /= scaleFactor;
		bb.ul.y /= scaleFactor;
		bb.lr.x /= scaleFactor;
		bb.lr.y /= scaleFactor;

		// add a couple of vertices at the end of the XGLVertexList
		// Used for drawing a ray from current indicatorIdx to right side of
		// bounding box for the glyph.
		v.push_back({ {0.0, 0.0, 0.0}, {}, {}, XGLColors::white });
		v.push_back({ {1.0, 1.0, 0.0}, {}, {}, XGLColors::white });

		// update this shape's VBO with new geometry from the CPU-side XGLVertexList
		// so it will actually be seen.
		Load(shader, v, idx);
	}

	void Draw() {
		int endIndex = contourOffsets[contourOffsets.size() - 1];
		XGLVertexAttributes *pData = v.data();
		int length = 2 * sizeof(XGLVertexAttributes);
		int endOffset = endIndex * sizeof(XGLVertexAttributes);

		if (showTail) {
			v[endIndex].v.x = v[indicatorIdx].v.x;
			v[endIndex].v.y = v[indicatorIdx].v.y;
			v[endIndex + 1].v.x = bb.lr.x;
			v[endIndex + 1].v.y = v[indicatorIdx].v.y;
			glBufferSubData(GL_ARRAY_BUFFER, endOffset, length, &pData[endIndex]);
		}

		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		if (contourOffsets.size()) {
			for (int idx = 0; idx < contourOffsets.size() - 1; idx++) {
				GLuint start = contourOffsets[idx];
				GLuint length = contourOffsets[idx + 1] - start;

				glDrawArrays(GL_LINE_LOOP, start, length);
				GL_CHECK("glDrawArrays() failed");
			}

			glPointSize(8.0);
			glDrawArrays(GL_POINTS, indicatorIdx, 1);
			glPointSize(1.0);

			if (showTail)
				glDrawArrays(GL_LINES, contourOffsets[contourOffsets.size() - 1], 2);
		}
		glDisable(GL_BLEND);
	}

	int indicatorIdx{ 0 };
	bool showTail = false;
	FT::BoundingBox bb;

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
};

static XGLFreeType *pFt;

void ExampleXGL::BuildScene() {
	XGLImGui *ig = nullptr;

	glm::vec3 cameraPosition(-1, -3, 5);
	glm::vec3 cameraDirection = glm::normalize(cameraPosition*-1.0f);
	glm::vec3 cameraUp = { 0, 0, 1 };
	camera.Set(cameraPosition, cameraDirection, cameraUp);

	AddShape("shaders/000-simple", [&](){ pFt = new XGLFreeType(); return pFt; });

	if ((ig = (XGLImGui *)(GetGuiManager()->FindObject("TitlerGui"))) != nullptr) {
		ig->AddMenuFunc([&]() {
			if (ImGui::Begin("Titler")) {
				if (ImGui::CollapsingHeader("Tweeks")) {
					ImGui::SliderInt("Indicator Index", &pFt->indicatorIdx, 0, pFt->v.size()-1);
					ImGui::Checkbox("Show Tail", &pFt->showTail);
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
