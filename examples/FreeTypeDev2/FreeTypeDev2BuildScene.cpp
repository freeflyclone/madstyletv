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
#include "XGLFreetypeUtils.h"

#include FT_OUTLINE_H

#ifdef FONT_NAME
#undef FONT_NAME
#endif

#define FONT_NAME "C:/windows/fonts/times.ttf"

class XGLFreeType : public FT::GlyphDecomposer,  public Triangulator {
public:
	typedef std::map<FT_ULong, FT_UInt> CharMap;
	struct ContourLayer {
		ContourLayer(int i) : idx(i) {};
		int idx;
		GLuint drawMode;
		std::vector<int> endOffsets;
	};

	typedef std::vector<ContourLayer*> ContourLayers;

	XGLFreeType(XGL* pxgl) : pXgl(pxgl) {
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

		// Force initial XGLBuffer::Load() to initialize the materials properties. (it won't with empty VertexAttributeList)
		v.push_back({});
	}

	void AdvanceGlyphPosition() {
		advance.x += face->glyph->advance.x / Triangulator::ScaleFactor();
		advance.y += face->glyph->advance.y / Triangulator::ScaleFactor();
	}

	void PushVertex(XGLVertexAttributes vrtx, bool isClockwise) {
		vrtx.v.x /= Triangulator::ScaleFactor();
		vrtx.v.y /= Triangulator::ScaleFactor();

		vrtx.v.x += advance.x;
		vrtx.v.y += advance.y;
		vrtx.v.z = 0.0001f;

		// this is part of the hybrid approach: help the shaders 
		// understand the glyphs's outline vs holes: holes get negative alpha
		if (!isClockwise)
			vrtx.c.a *= -1.0f;

		v.push_back(vrtx);
	}

	void EmitOutline() {
		ContourLayer *layer = CurrentLayer();

		// For each contour of the glyph...
		int contourIdx = 0;
		for (FT::Contour contour : Outline()) {
			xprintf("contourIdx: %d\n", contourIdx++);

			// determine if this contour is clockwise or counter-clockwise
			contour.ComputeCentroid();

			// add each contour vertex to this shape's CPU-side XGLVertexList
			for (XGLVertexAttributes vrtx : contour.v)
				PushVertex(vrtx, contour.isClockwise);

			//PushVertex(contour.v[0], contour.isClockwise);
			layer->endOffsets.push_back(v.size());
		}
	}

	void EmitTriangles() {
		for (FT::Contour contour : Outline()) {
			XGLVertexAttributes *vertices = contour.v.data();
			XGLVertexAttributes vrtx;
			std::size_t count = contour.v.size();

			contour.ComputeCentroid();

			for (std::size_t idx = 0; idx < count-2; idx += 2) {
				vrtx = vertices[idx];
				vrtx.t = { 0.0, 0.0 };
				PushVertex(vrtx, contour.isClockwise);

				vrtx = vertices[idx+1];
				vrtx.t = { 0.5, 0.0 };
				PushVertex(vrtx, contour.isClockwise);

				vrtx = vertices[idx+2];
				vrtx.t = { 1, 1 };
				PushVertex(vrtx, contour.isClockwise);
			}

			vrtx = vertices[count - 2];
			vrtx.t = { 0.0, 0.0 };
			PushVertex(vrtx, contour.isClockwise);

			vrtx = vertices[count - 1];
			vrtx.t = { 0.5, 0.0 };
			PushVertex(vrtx, contour.isClockwise);

			vrtx = vertices[0];
			vrtx.t = { 1.0, 1.0 };
			PushVertex(vrtx, contour.isClockwise);
		}
	}

	void RenderText() {
		v.clear();

		if (triangulateEnable) {
			ContourLayer *layer = NewContourLayer(v.size());
			advance = { 0.0f, 0.0f, 0.0f };

			// for each char in string...
			for (char c : renderString) {
				if (c == ' ') {
					AdvanceGlyphPosition();
					continue;
				}

				FT_Load_Glyph(face, charMap[c], FT_LOAD_FORCE_AUTOHINT | FT_LOAD_TARGET_NORMAL);

				FT::GlyphDecomposer::Reset();

				// This process results in a set of FT::Contours (single PSLG outlines) for the glyph
				FT_Outline_Decompose(&face->glyph->outline, (FT_Outline_Funcs*)this, (FT_Outline_Funcs*)this);

				// Convert the FT::Contours list to a triangulated mesh
				Triangulator::Convert(Outline(), advance);

				// mark end offset, so display loop can calculate size
				layer->endOffsets.push_back((int)v.size());
				AdvanceGlyphPosition();
			}
			layer->drawMode = GL_TRIANGLES;
		}

		if (outlineEnable) {
			ContourLayer *layer = NewContourLayer(v.size());
			advance = { 0.0f, 0.0f, 0.0f };

			// for each char in string...
			for (char c : renderString) {
				if (c == ' ') {
					AdvanceGlyphPosition();
					continue;
				}

				FT_Load_Glyph(face, charMap[c], FT_LOAD_FORCE_AUTOHINT | FT_LOAD_TARGET_NORMAL);

				FT::GlyphDecomposer::Reset();

				// This process results in a set of FT::Contours (single PSLG outlines) for the glyph
				FT_Outline_Decompose(&face->glyph->outline, (FT_Outline_Funcs*)this, (FT_Outline_Funcs*)this);
				EmitOutline();

				// mark end offset, so display loop can calculate size
				layer->endOffsets.push_back((int)v.size());

				AdvanceGlyphPosition();
			}
			layer->drawMode = GL_LINE_LOOP;
		}

		// update this shape's VBO with new geometry from the CPU-side XGLVertexList
		// so it will actually be seen.
		Load(shader, v, idx);

		return;
	}

	void RenderText(std::string textToRender) {
		renderString = textToRender;
		RenderText();
	}

	ContourLayer* CurrentLayer() {
		return contourLayers.back();
	}

	ContourLayer* NewContourLayer(int idx) {
		contourLayers.push_back(new ContourLayer(idx));
		return CurrentLayer();
	}

	void Draw() {
		for (ContourLayer* layer : contourLayers) {
			int start = layer->idx;
			if (layer->endOffsets.size()) {
				for (int i = 0; i < layer->endOffsets.size(); i++) {
					GLuint end = layer->endOffsets[i];
					GLuint length = end - start;
					glDrawArrays(layer->drawMode, start, length);
					GL_CHECK("glDrawArrays() failed");

					start = end;
				}
			}
		}
	}

	XGL* pXgl;

private:
	// These 2 numbers help Freetype math have adequate precision...
	const FT_F26Dot6 ftSize{ 1024 };
	const FT_UInt ftResolution{ 1024 };

	FT_Library ft;
	FT_Face face;
	FT_GlyphSlot g;
	CharMap charMap;
	XGLVertex advance;
	//std::vector<int>contourOffsets;
	std::string renderString;
	GLuint drawMode;
	bool triangulateEnable{ true };
	bool outlineEnable{ true };

	ContourLayers contourLayers;
};

static XGLFreeType *pFt;

void ExampleXGL::BuildScene() {
	XGLImGui *ig = nullptr;

	glm::vec3 cameraPosition(-1, -3, 5);
	glm::vec3 cameraDirection = glm::normalize(cameraPosition*-1.0f);
	glm::vec3 cameraUp = { 0, 0, 1 };
	camera.Set(cameraPosition, cameraDirection, cameraUp);

	AddShape("shaders/bezierPix", [&](){ pFt = new XGLFreeType(this); return pFt; });

	if ((ig = (XGLImGui *)(GetGuiManager()->FindObject("TitlerGui"))) != nullptr) {
		ig->AddMenuFunc([&]() {
			if (ImGui::Begin("Titler")) {
				if (ImGui::CollapsingHeader("Tweeks", ImGuiTreeNodeFlags_DefaultOpen)) {
				}
				ImGui::End();
			}
			else
				ImGui::End();

			return;
		});
	}

	//pFt->RenderText("&");
	pFt->RenderText("POOR THING");
}
