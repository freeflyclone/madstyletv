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
	typedef std::vector<GLsizei> ContourEndPoints;

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

	void RenderText() {
		v.clear();
		contourOffsets.clear();

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
			contourOffsets.push_back((int)v.size());
			AdvanceGlyphPosition();
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

	void Draw() {
		if (contourOffsets.size() > 1) {
			GLuint start = 0;
			GLuint end = contourOffsets[0];

			for (int i = 0; i < contourOffsets.size() - 1; i++) {
				GLuint length = end - start;

				glDrawArrays(GL_TRIANGLES, start, length);
				GL_CHECK("glDrawArrays() failed");

				start = contourOffsets[i];
				end = contourOffsets[i + 1];
			}

			GLuint length = v.size() - start;
			glDrawArrays(GL_TRIANGLES, start, length);
			GL_CHECK("glDrawArrays() failed");
		}
		else {
			if (v.size()){
				glDrawArrays(GL_TRIANGLES, 0, v.size());
				GL_CHECK("glDrawArrays() failed");
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
	std::vector<int>contourOffsets;
	std::string renderString;
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

	pFt->RenderText("&");
	pFt->RenderText("goop");
}
