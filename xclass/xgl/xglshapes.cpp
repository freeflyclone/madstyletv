#include "xgl.h"
XGLShape::XGLShape() {
	//DebugPrintf("XGLShape::XGLShape()\n");
	SetName("XGLShape");
}

XGLShape::~XGLShape(){
    //DebugPrintf("XGLShape::~XGLShape()\n");
}

void XGLShape::SetTheFunk(XGLShape::AnimaFunk fn){
    funk = fn;
}

void XGLShape::Animate(float clock){
    if (funk)
        funk(this, clock);
}

void XGLShape::Transform(glm::mat4 tm){
    model = tm;
}

void XGLShape::SetColor(XGLColor c) {
	XGLVertexAttributes *pv = MapVertexBuffer();
	for ( int i = 0; i < v.size(); i++)
		pv[i].c = c;
	UnmapVertexBuffer();
}

void XGLShape::Render(float clock) {
	glBufferSubData(GL_UNIFORM_BUFFER, sizeof(glm::mat4) * 2, sizeof(glm::mat4), (GLvoid *)&model);

	XGLBuffer::Bind();
	XGLMaterial::Bind(program);
	Animate(clock);
	Draw();
	Unbind();

	XGLObjectChildren children = Children();
	if (children.size()) {
		XGLObjectChildren::iterator ci;
		for (ci = children.begin(); ci != children.end(); ci++) {
			XGLShape* childShape = (XGLShape *)*ci;
			childShape->Render(model * childShape->model, clock);
		}
	}
}

void XGLShape::Render(glm::mat4 modelChain, float clock) {
	glBufferSubData(GL_UNIFORM_BUFFER, sizeof(glm::mat4) * 2, sizeof(glm::mat4), (GLvoid *)&modelChain);
	GL_CHECK("glBufferSubData() failed");

	XGLBuffer::Bind();
	XGLMaterial::Bind(program);
	Animate(clock);
	Draw();
	Unbind();

	XGLObjectChildren children = Children();
	if (children.size()) {
		XGLObjectChildren::iterator ci;
		for (ci = children.begin(); ci != children.end(); ci++) {
			XGLShape* childShape = (XGLShape *)*ci;
			childShape->Render(modelChain * childShape->model, clock);
		}
	}
}

XYPlaneGrid::XYPlaneGrid() {
	SetName("XYPlaneGrid");
	const float size = 100.0f;
	const int gridIncrement = 10;
	const XGLColor gridColor = { 0.5, 0, 0 };

	for (int i = 0; i <= 100; i += gridIncrement) {
		v.push_back({ { float(i), size, 0 }, {}, {}, gridColor });
		v.push_back({ { float(i), -size, 0 }, {}, {}, gridColor });
		v.push_back({ { -float(i), size, 0 }, {}, {}, gridColor });
		v.push_back({ { -float(i), -size, 0 }, {}, {}, gridColor });
		v.push_back({ { size, float(i), 0 }, {}, {}, gridColor });
		v.push_back({ { -size, float(i), 0 }, {}, {}, gridColor });
		v.push_back({ { size, -float(i), 0 }, {}, {}, gridColor });
		v.push_back({ { -size, -float(i), 0 }, {}, {}, gridColor });
	}

	//Load(v);
}

void XYPlaneGrid::Draw(){
	glDrawArrays(GL_LINES, 0, GLsizei(v.size()));
	GL_CHECK("glDrawArrays() failed");
}



XGLTriangle::XGLTriangle() {
    v.push_back({ { -1, 0, 0 }, {}, {}, { 1, 0, 0 } });
    v.push_back({ { 1, 0, 0 }, {}, {}, { 0, 1, 0 } });
    v.push_back({ { 0, 0, 1.412 }, {}, {}, { 0, 0, 1 } });

    //Load(v);
}
void XGLTriangle::Draw(){
    glDrawArrays(GL_TRIANGLES, 0, 3);
    GL_CHECK("glDrawArrays() failed");
};


XGLCube::XGLCube() {
	SetName("XGLCube");

	v.push_back({ { -1.0, -1.0, -1.0 }, {}, { -1.0, -1.0, -1.0 }, white });
	v.push_back({ { -1.0, 1.0, -1.0 }, {}, { -1.0, 1.0, -1.0 }, white });
	v.push_back({ { 1.0, -1.0, -1.0 }, {}, { 1.0, -1.0, -1.0 }, white });
	v.push_back({ { 1.0, 1.0, -1.0 }, {}, { 1.0, 1.0, -1.0 }, white });
	v.push_back({ { -1.0, -1.0, 1.0 }, {}, { -1.0, -1.0, 1.0 }, white });
	v.push_back({ { -1.0, 1.0, 1.0 }, {}, { -1.0, 1.0, 1.0 }, white });
	v.push_back({ { 1.0, -1.0, 1.0 }, {}, { 1.0, -1.0, 1.0 }, white });
	v.push_back({ { 1.0, 1.0, 1.0 }, {}, { 1.0, 1.0, 1.0 }, white });

	idx.push_back(0);
	idx.push_back(1);
	idx.push_back(2);
	idx.push_back(3);
	idx.push_back(7);
	idx.push_back(1);
	idx.push_back(5);
	idx.push_back(4);
	idx.push_back(7);
	idx.push_back(6);
	idx.push_back(2);
	idx.push_back(4);
	idx.push_back(0);
	idx.push_back(1);
}
void XGLCube::Draw() {
	glDrawElements(GL_TRIANGLE_STRIP, (GLsizei)(idx.size()), XGLIndexType, 0);
	GL_CHECK("glDrawElements() failed");
}


XGLSphere::XGLSphere(float r, int n) : radius(r), nSegments(n), visualizeNormals(false) {
	SetName("XGLSphere");
	int i, j;
	float twoPi = (2 * (float)PI);
	XGLVertexAttributes vrtx;

	for (j = 0; j <nSegments; j++) {
		for (i = 0; i < nSegments; i++) {
			float angle = (float)i * twoPi / (float)nSegments;
			float angle2 = (float)j * twoPi / (float)nSegments;

			float x = sin(angle)*cos(angle2) * radius;
			float y = cos(angle) * radius;
			float z = (sin(angle)*sin(angle2)) * radius;

			vrtx.v = { x,y,z };
			vrtx.c = white;
			vrtx.n = { x / radius, y / radius, z / radius };

			v.push_back(vrtx);
		}
	}

	int nVerts = (int)v.size();

	idx.push_back(1);
	for (i = 0; i <= nVerts; i++)
		idx.push_back(((i & 1) == 0) ? (i >> 1) % (XGLIndex)(v.size()) : ((i >> 1) + nSegments) % (XGLIndex)(v.size()));

	if (visualizeNormals) {
		for (i = 0; i < nVerts; i++) {
			XGLVertexAttributes nVrtx = v[i];
			vrtx = nVrtx;
			vrtx.v /= radius;
			vrtx.v *= 2;

			nVrtx.v.x += vrtx.v.x;
			nVrtx.v.y += vrtx.v.y;
			nVrtx.v.z += vrtx.v.z;

			v.push_back(nVrtx);
		}
		for (i = 0; i < nVerts; i++) {
			idx.push_back(i%nVerts);
			idx.push_back((i%nVerts) + nVerts);
		}
	}
}

void XGLSphere::Draw() {
	glDrawElements(GL_TRIANGLE_STRIP, (GLsizei)(idx.size()), XGLIndexType, 0);
	//glDrawElements(GL_LINES, (GLsizei)(idx.size()), XGLIndexType, 0);
	GL_CHECK("glDrawElements() failed");
}


// draw 2 hemispheres, joined by a cylinder, centered about the origin, along the X axis
// (This is how PhysX likes it)
XGLCapsule::XGLCapsule(float r, float l, int n) : radius(r), length(l), nSegments(n) {
	SetName("XGLCapsule");
	int i, j;
	float twoPi = (2 * (float)PI);
	float pi = (float)PI;
	XGLVertexAttributes vrtx;
	float angleStep = pi / float(nSegments);
	float angleStep2 = pi / float(nSegments);
	int totalSegments = nSegments * 2;
	int nPointsSemi = (nSegments / 2) + 1;
	int nPoints = nSegments + 2;

	for (j = 0; j < totalSegments; j++) {
		for (i = 0; i < nPointsSemi; i++) {
			float angle = (float)i * angleStep;
			float angle2 = (float)j * angleStep2;

			float x = (cos(angle) * radius) + length / 2.0f;
			float y = -sin(angle)*cos(angle2) * radius;
			float z = (sin(angle)*sin(angle2)) * radius;

			vrtx.v = { x,y,z };
			vrtx.c = white;
			vrtx.n = { x / radius, y / radius, z / radius };
			v.push_back(vrtx);
		}
		// The "bottom" half. Overlaps with "top" half by one point, on purpose,
		// so "radius" will be respected along the cylindrical portion
		for (i = nPointsSemi - 1; i >= 0; i--) {
			float angle = (float)i * angleStep;
			float angle2 = (float)j * angleStep2;

			float x = -1.0f * (cos(angle) * radius) - length / 2.0f;
			float y = -sin(angle)*cos(angle2) * radius;
			float z = (sin(angle)*sin(angle2)) * radius;

			vrtx.v = { x,y,z };
			vrtx.c = white;
			vrtx.n = { x / radius, y / radius, z / radius };
			v.push_back(vrtx);
		}
	}

	int nVerts = (int)v.size();
	int nextIdx;
	int mostRecentEvenIdx;
	for (j = 0; j < totalSegments * 2; j++) {
		if ((j & 1) == 0) {
			for (i = 0; i < nPoints * 2; i++)
			{
				if ((i & 1) == 0)
					nextIdx = (j*nPoints) + i / 2;
				else
					nextIdx = ((j + 1)*nPoints) + i / 2;

				nextIdx %= nVerts;
				idx.push_back(nextIdx);
			}
			idx.push_back(nextIdx);
		}
		else {
			for (i = 0; i < nPoints * 2; i++)
			{
				if ((i & 1) == 0) {
					nextIdx = ((j + 2)*nPoints - 1) - i / 2;
					mostRecentEvenIdx = nextIdx;
				}
				else
					nextIdx = ((j + 1)*nPoints - 1) - i / 2;

				nextIdx %= nVerts;
				idx.push_back(nextIdx);
			}
			mostRecentEvenIdx %= nVerts;
			idx.push_back(mostRecentEvenIdx);
		}
	}
}

XGLCapsule::~XGLCapsule() {
}

void XGLCapsule::Draw() {
	glDrawElements(GL_TRIANGLE_STRIP, (GLsizei)(idx.size()), XGLIndexType, 0);
	GL_CHECK("glDrawElements() failed");
}
// A torus can be thought of as a "minor circle" that is coplanar to an axis
// of revolution, rotated about that axis in a "major circle".  Thus to specify
// a Torus one needs only specify the radii of the two circles, and the number
// of line segments comprising each.  This is fairly classic 3D graphics
// knowledge indeed, I copied the actual formulae from a web site.
XGLTorus::XGLTorus(float rMaj, float rMin, int nMaj, int nMin) :
    radiusMajor(rMaj),
    radiusMinor(rMin),
    nSegmentsMajor(nMaj),
    nSegmentsMinor(nMin),
    visualizeNormals(false)
{
	SetName("XGLTorus");
    XGLVertexAttributes vrtx;
    
    float twoPi = (2 * (float)PI);
    XGLIndex nMajor = nSegmentsMajor;
    XGLIndex nMinor = nSegmentsMinor;
    int i, j;
    float s, t;
    float iAngle, jAngle;
    float tx, ty, tz;
    float sx, sy, sz;
    float nx, ny, nz;
    //float len;

	// The major circle
    for (j = 0; j < nMajor; j++) {
		t = float(j % nMajor);
        jAngle = t * twoPi / nMajor;

		// The minor circle
        for (i = 0; i < nMinor; i++) {
			s = (i) % nMinor + 0.5f;
            iAngle = s * twoPi / nMinor;

			// the 3D coordinate of this vertex
            float x = (radiusMajor + radiusMinor*cos(iAngle))  * cos(jAngle);
            float y = (radiusMajor + radiusMinor*cos(iAngle))  * sin(jAngle);
            float z = radiusMinor * sin(iAngle);

			// "normal" generation...
			// the major circle component of the normal vector
            tx = -sin(jAngle);
            ty = cos(jAngle);
            tz = 0;

			// the minor circle component of the normal vector
            sx = cos(jAngle)*(-sin(iAngle));
            sy = sin(jAngle)*(-sin(iAngle));
            sz = cos(iAngle);

			// vector multiplaction for the actual normal vector
            nx = ty*sz - tz*sy;
            ny = tz*sx - tx*sz;
            nz = tx*sy - ty*sx;

            //len = sqrt(nx*nx + ny*ny + nz*nz);
 
            vrtx.v = { x, y, z };
            vrtx.n = { nx, ny, nz };
            vrtx.t = { 0.0f, 0.0f };
            vrtx.c = { 1, 1, 1 };
            v.push_back(vrtx);
        }
    }

	// Build an index buffer so we can use GL_TRIANGLE_STRIP...
	// The vertex positions are specified above as corners of a quadrilateral,
	// and it takes 2 triangles per quad...
	for (XGLIndex n = 0; n <= v.size() * 2; n++) {
		int index;

		// to build the index list, we alternate back and forth between
		// two "minor circles", one "on the left" and one "on the right"
		// Even values of "n" are on the left, and and odd are on the right.
		// Thankfully, with a torus this pattern automagically wraps from
		// one pair of minor circles to the next, so this just works all the
		// way around the torus "major circle".
		if ((n & 1) == 0)
			index = (n >> 1) % (XGLIndex)(v.size());
		else
			index = ((n >> 1) + nSegmentsMinor) % (XGLIndex)(v.size());

		idx.push_back(index);
	}
    
	// Final compensation for the fact that a triangle strip requires an initial 2
	// points before "point-per-triangle" kicks in.  Initial compensation is
	// that the loop above iterates with "less than or equal to"
	idx.push_back(nSegmentsMinor);

    nTorusIndices = (GLsizei)(idx.size());

    if (visualizeNormals) 
    {
        // normals visualization
        int nv = static_cast<int>(v.size());
        xprintf("There are %d vertices\n", nv);

        for (int i = 0; i < nv; i++){
            XGLVertexAttributes tv = { { v[i].v.x, v[i].v.y, v[i].v.z }, {}, {}, white };
            XGLVertexAttributes tn = { { v[i].n.x, v[i].n.y, v[i].n.z }, {}, {}, white };
            XGLVertexAttributes tvn;

            tvn.v.x = tv.v.x + tn.v.x;
            tvn.v.y = tv.v.y + tn.v.y;
            tvn.v.z = tv.v.z + tn.v.z;

			tvn.c.r = tv.c.r;
			tvn.c.g = tv.c.g;
			tvn.c.b = tv.c.b;

			v.push_back(tvn);
            idx.push_back(i);
            idx.push_back(i + nv);
        }
    }

    nTotalIndices = (GLsizei)(idx.size());
}

void XGLTorus::Draw() {
    //glDrawRangeElements(GL_TRIANGLE_STRIP, 0, nTotalIndices, nTorusIndices, XGLIndexType, 0);
	glDrawElements(GL_TRIANGLE_STRIP, (GLsizei)(idx.size()), XGLIndexType, 0);
	GL_CHECK("glDrawRangeElements() failed");
	/*
    if (visualizeNormals){
        glDrawRangeElements(GL_LINES, nTorusIndices, nTotalIndices, nTotalIndices, XGLIndexType, 0);
        GL_CHECK("glDrawRangeElements failed()");
    }
	*/
}

XGLIcoSphere::XGLIcoSphere() {
	SetName("XGLIcoSphere");
	float t = (float)sqrt(2.0) / 2.0f;

	v.push_back({ { -t,  t, 0 },{},{}, red });
	v.push_back({ { t,  t, 0 },{},{}, red });
	v.push_back({ { -t, -t, 0 },{},{}, red });
	v.push_back({ { t, -t, 0 },{},{}, red });

	v.push_back({ { 0, -t,  t },{},{}, green });
	v.push_back({ { 0,  t,  t },{},{}, green });
	v.push_back({ { 0, -t, -t },{},{}, green });
	v.push_back({ { 0,  t, -t },{},{}, green });

	v.push_back({ {  t, 0, -t },{},{}, blue });
	v.push_back({ {  t, 0,  t },{},{}, blue });
	v.push_back({ { -t, 0, -t },{},{}, blue });
	v.push_back({ { -t, 0,  t },{},{}, blue });


	idx.push_back(0);	idx.push_back(11);	idx.push_back(5);
	idx.push_back(0);	idx.push_back(5);	idx.push_back(1);
	idx.push_back(0);	idx.push_back(1);	idx.push_back(7);
	idx.push_back(0);	idx.push_back(7);	idx.push_back(10);
	idx.push_back(0);	idx.push_back(10);	idx.push_back(11);

	idx.push_back(1);	idx.push_back(5);	idx.push_back(9);
	idx.push_back(5);	idx.push_back(11);	idx.push_back(4);
	idx.push_back(11);	idx.push_back(10);	idx.push_back(2);
	idx.push_back(10);	idx.push_back(7);	idx.push_back(6);
	idx.push_back(7);	idx.push_back(1);	idx.push_back(8);

	idx.push_back(3);	idx.push_back(9);	idx.push_back(4);
	idx.push_back(3);	idx.push_back(4);	idx.push_back(2);
	idx.push_back(3);	idx.push_back(2);	idx.push_back(6);
	idx.push_back(3);	idx.push_back(6);	idx.push_back(8);
	idx.push_back(3);	idx.push_back(8);	idx.push_back(9);

	idx.push_back(4);	idx.push_back(9);	idx.push_back(5);
	idx.push_back(2);	idx.push_back(4);	idx.push_back(11);
	idx.push_back(6);	idx.push_back(2);	idx.push_back(10);
	idx.push_back(8);	idx.push_back(6);	idx.push_back(7);
	idx.push_back(9);	idx.push_back(8);	idx.push_back(1);
}

void XGLIcoSphere::Draw() {
	glDrawElements(GL_TRIANGLES, (GLsizei)(idx.size()), XGLIndexType, 0);
	GL_CHECK("glDrawElements() failed");
}

XGLSphere2::XGLSphere2(float r, int n) : radius(r), nSegments(n), visualizeNormals(false) {
	SetName("XGLSphere2_");
	int i, j;
	float twoPi = (2 * (float)PI);
	float pi = (float)PI;
	XGLVertexAttributes vrtx;
	int nPoints = nSegments+1;
	float angleStep = pi / float(nSegments);
	float angleStep2 = pi / float(nSegments);

	for (j = 0; j < nSegments*2; j++) {
		for (i = 0; i < nPoints; i++) {
			float angle = (float)i * angleStep;
			float angle2 = (float)j * angleStep2;

			float x = -sin(angle)*cos(angle2) * radius;
			float y = cos(angle) * radius;
			float z = (sin(angle)*sin(angle2)) * radius;

			vrtx.v = { x,y,z };
			vrtx.c = white;
			vrtx.n = { x / radius, y / radius, z / radius };

			v.push_back(vrtx);
		}
	}

	int nVerts = (int)v.size();
	int nextIdx;
	int mostRecentEvenIdx;
	for (j = 0; j < nSegments*2; j++) {
		if ((j & 1) == 0) {
			for (i = 0; i < nPoints * 2; i++)
			{
				if ((i & 1) == 0)
					nextIdx = (j*nPoints) + i / 2;
				else
					nextIdx = ((j+1)*nPoints) + i / 2;

				nextIdx %= nVerts;
				idx.push_back(nextIdx);
			}
			idx.push_back(nextIdx);
		}
		else {
			for (i = 0; i < nPoints * 2; i++)
			{
				if ((i & 1) == 0) {
					nextIdx = ((j + 2)*nPoints - 1) - i / 2;
					mostRecentEvenIdx = nextIdx;
				}
				else
					nextIdx = ((j + 1)*nPoints - 1) - i / 2;

				nextIdx %= nVerts;
				idx.push_back(nextIdx);
			}
			mostRecentEvenIdx %= nVerts;
			idx.push_back(mostRecentEvenIdx);
		}
	}
}

void XGLSphere2::Draw() {
	glDrawArrays(GL_POINTS, 0, GLsizei(v.size()));
	GL_CHECK("glDrawArrays() failed");
	glDrawElements(GL_TRIANGLE_STRIP, (GLsizei)(idx.size()), XGLIndexType, 0);
	GL_CHECK("glDrawElements() failed");
}

XGLTextureAtlas::XGLTextureAtlas() {
	SetName("XGLTextureAtlas");
	XGLColor white = { 1, 1, 0 };

	gridCellWidth = 10.0f;
	gridCellHeight = 10.0f;

	float halfWidth = gridCellWidth / 2.0f;
	float halfHeight = gridCellHeight / 2.0f;

	int atlasPagesDiv2 = (font.atlasPageCount + 1) >> 1;
	int factor;

	for (factor = atlasPagesDiv2 - 1; factor >= 2; factor--)
		if ((atlasPagesDiv2 % factor) == 0)
			break;

	if (factor == 0)
		factor = 1;

	gridXsize = (font.atlasPageCount + 1) / factor;
	gridYsize = factor;

	if (gridYsize > gridXsize) {
		gridYsize = gridXsize;
		gridYsize = factor;
	}

	int upperLeftX = -(gridXsize + 1) / 2;
	int upperLeftY = (gridYsize + 1) / 2;

	int i = 0;
	for (int y = 0; y < gridYsize; y++) {
		for (int x = 0; x < gridXsize; x++) {
			float xPos = ((upperLeftX + x)*gridCellWidth) + halfWidth;
			float yPos = ((upperLeftY - y)*gridCellHeight) - halfHeight;
			float xScale = 1.0f;
			float yScale = 1.0f;

			if (font.atlasHeight > font.atlasWidth)
				xScale *= (float)font.atlasWidth / (float)font.atlasHeight;
			else
				yScale *= (float)font.atlasHeight / (float)font.atlasWidth;

			v.push_back({ { xPos - halfWidth*xScale, yPos - halfHeight*yScale, 0 }, { 0, 1 }, {}, white });
			v.push_back({ { xPos - halfWidth*xScale, yPos + halfHeight*yScale, 0 }, { 0, 0 }, {}, white });
			v.push_back({ { xPos + halfWidth*xScale, yPos - halfHeight*yScale, 0 }, { 1, 1 }, {}, white });
			v.push_back({ { xPos + halfWidth*xScale, yPos + halfHeight*yScale, 0 }, { 1, 0 }, {}, white });

			int index = i * 4;

			idx.push_back(index + 0);
			idx.push_back(index + 1);
			idx.push_back(index + 2);
			idx.push_back(index + 3);

			AddTexture(FONT_NAME, font.atlasWidth, font.atlasHeight, 1, font.bitmapPages[i]);
			i++;
			if (i == font.atlasPageCount)
				break;
		}
		if (i == font.atlasPageCount)
			break;
	}
}

void XGLTextureAtlas::Draw() {
	glEnable(GL_BLEND);
	GL_CHECK("glEnable(GL_BLEND) failed");
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	GL_CHECK("glBlendFunc() failed");

	for (unsigned int i = 0; i < font.atlasPageCount; i++) {
		glBindTexture(GL_TEXTURE_2D, texIds[i]);
		GL_CHECK("glBindTexture() failed");
		glDrawElementsBaseVertex(GL_TRIANGLE_STRIP, 4, XGLIndexType, 0, i * 4);
		GL_CHECK("glDrawElements() failed");
	}

	glDisable(GL_BLEND);
	GL_CHECK("glDisable(GL_BLEND) failed");
}

XGLTexQuad::XGLTexQuad(std::string fileName) {
	SetName("XTexQuad");
	const XGLColor white = { 1, 1, 1 };

	v.push_back({ { -1.0, -1.0, 0 }, { 0, 1 }, {}, white });
	v.push_back({ { -1.0, 1.0, 0 }, { 0, 0 }, {}, white });
	v.push_back({ { 1.0, -1.0, 0 }, { 1, 1 }, {}, white });
	v.push_back({ { 1.0, 1.0, 0 }, { 1, 0 }, {}, white });

	idx.push_back(0);
	idx.push_back(1);
	idx.push_back(2);
	idx.push_back(3);

	AddTexture(fileName);
}

XGLTexQuad::XGLTexQuad(std::string texName, int width, int height, int channels, GLubyte *img, bool flipColors) {
	SetName("XTexQuad");
	const XGLColor white = { 1, 1, 1 };

	v.push_back({ { -1.0, -1.0, 0 }, { 0, 1 }, {}, white });
	v.push_back({ { -1.0, 1.0, 0 }, { 0, 0 }, {}, white });
	v.push_back({ { 1.0, -1.0, 0 }, { 1, 1 }, {}, white });
	v.push_back({ { 1.0, 1.0, 0 }, { 1, 0 }, {}, white });

	idx.push_back(0);
	idx.push_back(1);
	idx.push_back(2);
	idx.push_back(3);

	AddTexture(texName, width, height, channels, img, flipColors);
}

void XGLTexQuad::Draw() {
	glEnable(GL_BLEND);
	GL_CHECK("glEnable(GL_BLEND) failed");
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	GL_CHECK("glBlendFunc() failed");

	glDrawElements(GL_TRIANGLE_STRIP, (GLsizei)(idx.size()), XGLIndexType, 0);
	GL_CHECK("glDrawElements() failed");

	glDisable(GL_BLEND);
	GL_CHECK("glDisable(GL_BLEND) failed");
}
