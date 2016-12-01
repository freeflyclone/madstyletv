#include "xgl.h"
#include <sstream>
#include <iomanip>
#include <iterator>

static void DebugChar(FT_Face face, int c){
    FT_Load_Char(face, c, FT_LOAD_RENDER);
    FT_GlyphSlot g = face->glyph;

    int width = g->bitmap.width;
    int height = g->bitmap.rows;
    int top = g->bitmap_top;
    int left = g->bitmap_left;
    int ax = g->advance.x;
    int ay = g->advance.y;

	xprintf("width: %d, height: %d\n", width, height);
	xprintf("left: %d, top: %d\n", left, top);
	xprintf("ax: %d, ay: %d\n", ax, ay);

    std::stringstream ss;
    ss << std::hex;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++)
            ss << std::setw(2) << std::setfill('0') << (int)(g->bitmap.buffer[y*width + x]);
        ss << std::endl;
    }
    
    std::string out = ss.str();

	xprintf("%s\n", out.c_str());
}

static void DebugString(FT_Face face, std::string c){
    for(unsigned int i=0; i<c.size(); i++)
        DebugChar(face, c.c_str()[i]);
}

XGLFont::XGLFont() : atlasWidth(0), atlasHeight(0), atlasPageCount(0), bitmapPages(NULL), maxAscend(0), maxDescend(0) {
    if (FT_Init_FreeType(&ft))
        throwXGLException("Init of FreeType failed");

    if (FT_New_Face(ft, FONT_NAME, 0, &face))
        throwXGLException("FT_New_Face() failed " FONT_NAME);

    if (FT_Select_Charmap(face, ft_encoding_unicode))
        throwXGLException("FT_Select_Charmap(UNICODE) failed.");

    // scale the rendering to 1 pixel per 'point' resolution
    FT_Set_Pixel_Sizes(face, 0, 64);

	FT_GlyphSlot g = face->glyph;

	//if (false)
	{
		FT_ULong charcode;
		FT_UInt gindex;
		XGLGlyph xGlyph;

		// build an XGLCharMap of the entire set of glyphs for this font.
		// (this could be huge for Chinese fonts)
		for (charcode = FT_Get_First_Char(face, &gindex); gindex; charcode = FT_Get_Next_Char(face, charcode, &gindex))
			charMap.emplace(charcode, gindex);

		const int glyphsPerRow = 32;
		const int rowsPerBitmap = 32;

		// face->num_glyphs isn't necessarily accurate
		const int numGlyphs = (const int)(charMap.size());
		const int glyphsPerBitmap = glyphsPerRow * rowsPerBitmap;
		const int numBitmaps = numGlyphs / glyphsPerBitmap + 1;

		int maxRowHeight = 0;
		int maxRowWidth = 0;

		xprintf("There are %d glyphs.\n", numGlyphs);

		XGLFont::CharMap::iterator it = charMap.begin();

		// iterate throw the entire charmap...
		// one "glyph row" at a time...
		// measure the total width of the row...
		// and set maxRowWidth and maxRowHeight...
		// so we can determine the size of an atlas rectangle.
		// (no, there's no optimal packing going on here)
		for (unsigned int i = 0; i < charMap.size(); i++, it++) {
			int rowHeight = 0;
			int rowWidth = 0;
			for (int j = 0; j < glyphsPerRow && it != charMap.end(); j++, it++) {
				FT_Load_Glyph(face, it->second, FT_LOAD_RENDER);
				if (g->metrics.height / 64 > rowHeight)
					rowHeight = g->metrics.height / 64;
				rowWidth += g->metrics.width / 64;

				int h = g->bitmap.rows;
				int t = g->bitmap_top;
				if ((h - t)>maxDescend)
					maxDescend = h - t;
				if (t > maxAscend)
					maxAscend = t;
			}
			if (it == charMap.end())
				break;
			if (rowWidth > maxRowWidth)
				maxRowWidth = rowWidth;
			if (rowHeight > maxRowHeight)
				maxRowHeight = rowHeight;
		}

		// now we know how tall a texture atlas needs to be
		const int bitmapHeight = maxRowHeight * rowsPerBitmap;

		// we already figured how many texture atlas bitmaps we'll need for the entire
		// glyph set, so allocate a set of pointers to those bitmap pages
		if ((bitmapPages = (GLubyte **)malloc(numBitmaps*sizeof(GLubyte *))) == NULL)
			throwXGLException("malloc() failed allocating list of bitmap pages");

		memset(bitmapPages, 0, sizeof(GLubyte *)*numBitmaps);

		// expose the dimensions we've calculated
		atlasWidth = maxRowWidth;
		atlasHeight = bitmapHeight;
		atlasPageCount = numBitmaps;

		//rewind the iterator
		it = charMap.begin();

		// fill up all alocated pages with glyph images.
		for (int l = 0; l < numBitmaps; l++) {
			if ((bitmapPages[l] = (GLubyte *)malloc(maxRowWidth*bitmapHeight)) == NULL)
				throwXGLException("malloc() failed allocating a texture atlas page");

			memset(bitmapPages[l], 0, maxRowWidth*bitmapHeight);

			// set "dest" once per page, it gets incremented vertically by the
			// size in bytes of a glyph row, every row.
			GLubyte *dest = bitmapPages[l];

			// fill up all rows...
			for (int k = 0; k < rowsPerBitmap; k++){
				int xOffset = 0;
				// fill up this row with its set of glyphs
				for (int j = 0; j < glyphsPerRow && it != charMap.end(); j++, it++){
					// get the glyph
					FT_Load_Glyph(face, it->second, FT_LOAD_RENDER);

					// and blit it to the this texture atlas page at the current 
					// "xOffset" and current glyph row...
					// (this is classic 2D blit C code.)
					GLubyte *src = (GLubyte *)g->bitmap.buffer;
					for (unsigned int i = 0; i < g->bitmap.rows; i++) {
						memcpy((dest + (i*atlasWidth) + (xOffset)), src, g->bitmap.width);
						src += g->bitmap.width;
					}

					// Start: where XGLGlyph should be built and appended to glyphMap
					xGlyph.index = it->second;
					xGlyph.page = l;
					xGlyph.row = k;
					xGlyph.xOff = xOffset;
					xGlyph.yOff = k * maxRowHeight;
					xGlyph.width = g->bitmap.width;
					xGlyph.height = g->bitmap.rows;
					glyphMap.emplace(it->first, xGlyph);
					// End: adding XGLGlyph to glyphMap

					xOffset += g->advance.x / 64;
				}
				dest += maxRowWidth * maxRowHeight;
				if (it == charMap.end())
					break;
			}
			if (it == charMap.end())
				break;
		}
		// at this point there should be a 1:1 between charMap and glyphMap.
		XGLFont::CharMap::iterator cit = charMap.begin();
		XGLFont::GlyphMap::iterator git = glyphMap.begin();	
		for (; git != glyphMap.end(); cit++, git++){
			XGLGlyph xg = git->second;
			FT_UInt index = cit->second;
			if (xg.index != index)
				xprintf("MisMatch: %d vs %d\n", index, xg.index);
		}		
	}
}

XGLFont::~XGLFont(){
    FT_Done_Face(face);
    FT_Done_FreeType(ft);

    for (unsigned int i = 0; i < atlasPageCount; i++) {
        if (bitmapPages[i])
            free(bitmapPages[i]);
    }

    free(bitmapPages);
}

void XGLFont::SetPixelSize(int size) {
	pixelSize = size;
	FT_Set_Pixel_Sizes(face, 0, pixelSize);
}

void XGLFont::RenderText(std::string text, unsigned char *buffer, int width, int height, int *penX, int *penY) {
	FT_GlyphSlot g = font.face->glyph;
	GLubyte *src, *dest;
	int numGlyphs = (int)text.size();

	if (buffer == NULL)
		return;

	// Render the string...
	for (int i = 0; i < numGlyphs; i++){
		if (text[i] == '\n') {
			*penX = 10;
			*penY += pixelSize + (pixelSize / 16);
		}
		else {
			FT_Load_Glyph(font.face, font.charMap[text[i]], FT_LOAD_RENDER);
			dest = buffer + (*penY - g->bitmap_top) * width + *penX + g->bitmap_left;
			src = (GLubyte *)g->bitmap.buffer;

			// 2D blit the glyph into the texture at penX,penY
			for (unsigned int i = 0; i < g->bitmap.rows; i++) {
				memcpy(dest + (i*width), src, g->bitmap.width);
				src += g->bitmap.width;
			}
			*penX += (int)((float)g->advance.x / 64.0f + 0.5);
		}
	}
}

void XGLFont::RenderText(std::wstring text, unsigned char *buffer, int width, int height, int *penX, int *penY) {
	FT_GlyphSlot g = font.face->glyph;
	GLubyte *src, *dest;
	int numGlyphs = (int)text.size();

	if (buffer == NULL)
		return;

	// Render the string...
	for (int i = 0; i < numGlyphs; i++){
		if (text[i] == L'\n') {
			*penX = 10;
			*penY += pixelSize + (pixelSize / 16);
		}
		else {
			FT_Load_Glyph(font.face, font.charMap[text[i]], FT_LOAD_RENDER);
			dest = buffer + (*penY - g->bitmap_top) * width + *penX + g->bitmap_left;
			src = (GLubyte *)g->bitmap.buffer;

			// 2D blit the glyph into the texture at penX,penY
			for (unsigned int i = 0; i < g->bitmap.rows; i++) {
				memcpy(dest + (i*width), src, g->bitmap.width);
				src += g->bitmap.width;
			}
			*penX += (int)((float)g->advance.x / 64.0f + 0.5);
		}
	}
}
