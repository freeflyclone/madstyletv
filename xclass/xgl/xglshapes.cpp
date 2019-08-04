#include "xgl.h"


XGLShape::XGLShape() : isVisible(true), parent(nullptr) {
	//xprintf("XGLShape::XGLShape()\n");
	SetName("XGLShape");
}

XGLShape::~XGLShape(){
    //xprintf("XGLShape::~XGLShape(%s) %s children\n", Name().c_str(), Children().size()?"has":"does not have");
	if (Children().size()) {
		XObjectChildren children = Children();
		XObjectChildren::iterator ci;
		for (ci = children.begin(); ci < children.end(); ci++) {
			XGLShape *child = (XGLShape *)*ci;
			delete child;
		}
		children.clear();
	}
}

void XGLShape::AddChild(XGLShape *s) {
	s->parent = this;
	XObject::AddChild(s);
}

void XGLShape::SetAnimationFunction(XGLShape::AnimationFn fn){
    animationFunction = fn;
}

void XGLShape::Animate(float clock){
	if (animationFunction)
		animationFunction(clock);

	for (auto child : Children()) {
		if (dynamic_cast<XGLShape *>(child)) {
			XGLShape *childShape = (XGLShape *)child;
			childShape->Animate(clock);
		}
	}
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

void XGLShape::Render() {
	Render(model);
}

void XGLShape::Render(glm::mat4 modelChain) {
	if (v.size() > 0) {
		glProgramUniformMatrix4fv(shader->programId, shader->modelUniformLocation, 1, false, (GLfloat *)&modelChain);
		GL_CHECK("glProgramUniformMatrix4fv() failed");

		XGLBuffer::Bind();
		XGLMaterial::Bind(shader->programId);
	}
	Draw();

	if (v.size()>0)
		Unbind();

	for (auto child : Children()) {
		if (dynamic_cast<XGLShape *>(child)) {
			XGLShape *childShape = (XGLShape *)child;
			if (childShape->isVisible) {
				if (childShape->preRenderFunction)
					childShape->preRenderFunction(clock);

				childShape->Render(modelChain * childShape->model);

				if (childShape->postRenderFunction)
					childShape->postRenderFunction(clock);
			}
		}
	}
}

XGLAxis::XGLAxis(XGLVertex vertex, float length, XGLColor color) {
	SetName("XGLAxis");

	v.push_back({ glm::vec3(0), {}, {}, color });
	v.push_back({ vertex * length, {}, {}, color });
}

void XGLAxis::Draw(){
	glDrawArrays(GL_LINES, 0, GLsizei(v.size()));
	GL_CHECK("glDrawArrays() failed");
}

XYPlaneGrid::XYPlaneGrid(float size, float step) {
	SetName("XYPlaneGrid");
	const XGLColor gridColor = { 0.5, 0, 0, 1 };

	for (float i = 0; i <= 100.0f; i += step) {
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
	SetName("XGLTriangle");
	v.push_back({ { -1, 0, 0 }, {}, {}, { 1, 0, 0, 1 } });
	v.push_back({ { 1, 0, 0 }, {}, {}, { 0, 1, 0, 1 } });
	v.push_back({ { 0, 0, 1.412 }, {}, {}, { 0, 0, 1, 1 } });

    //Load(v);
}
void XGLTriangle::Draw(){
    glDrawArrays(GL_TRIANGLES, 0, 3);
    GL_CHECK("glDrawArrays() failed");
};


XGLCube::XGLCube() {
	SetName("XGLCube");

	v.push_back({ { -1.0, -1.0, -1.0 }, {}, { -1.0, -1.0, -1.0 }, XGLColors::white });
	v.push_back({ { -1.0, 1.0, -1.0 }, {}, { -1.0, 1.0, -1.0 }, XGLColors::white });
	v.push_back({ { 1.0, -1.0, -1.0 }, {}, { 1.0, -1.0, -1.0 }, XGLColors::white });
	v.push_back({ { 1.0, 1.0, -1.0 }, {}, { 1.0, 1.0, -1.0 }, XGLColors::white });
	v.push_back({ { -1.0, -1.0, 1.0 }, {}, { -1.0, -1.0, 1.0 }, XGLColors::white });
	v.push_back({ { -1.0, 1.0, 1.0 }, {}, { -1.0, 1.0, 1.0 }, XGLColors::white });
	v.push_back({ { 1.0, -1.0, 1.0 }, {}, { 1.0, -1.0, 1.0 }, XGLColors::white });
	v.push_back({ { 1.0, 1.0, 1.0 }, {}, { 1.0, 1.0, 1.0 }, XGLColors::white });

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


XGLSphere::XGLSphere(float r, int n) : radius(r), nSegments(n - (n&1)), visualizeNormals(false) {
	SetName("XGLSphere");
	int i, j;
	float twoPi = (2 * (float)PI);
	XGLVertexAttributes vrtx;
	int halfSeg = nSegments >> 1;
	int halfSegPlus = halfSeg + 1;

	for (j = 0; j <nSegments; j++) {
		for (i = 0; i < halfSegPlus; i++) {
			float angle = (float)i * twoPi / (float)nSegments;
			float angle2 = (float)j * twoPi / (float)nSegments;

			float x = sin(angle)*cos(angle2) * radius;
			float y = cos(angle) * radius;
			float z = (sin(angle)*sin(angle2)) * radius;

			vrtx.v = { x,y,z };
			vrtx.c = XGLColors::white;
			vrtx.n = { x / radius, y / radius, z / radius };

			v.push_back(vrtx);
		}
	}

	int nVerts = (int)v.size();

	j = 0;
	int count = (nSegments - 2);
	for (j = 0; j < count; j++) {
		for (int i = 0; i < halfSegPlus; i++) {
			idx.push_back((j*halfSegPlus) + i);
			idx.push_back(((j + 1)*halfSegPlus) + i);
		}

		for (int i = 1; i < halfSegPlus; i++) {
			idx.push_back(((j + 2)*halfSegPlus) + halfSegPlus - i);
			idx.push_back(((j + 1)*halfSegPlus) + halfSegPlus - i);
		}
	}
	for (int i = 0; i < halfSegPlus; i++) {
		idx.push_back(((j+1)*halfSegPlus) + i);
		idx.push_back(i);
	}

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
	GL_CHECK("glDrawElements() failed");
}


XGLHemiSphere::XGLHemiSphere(float r, int n) : radius(r), nSegments(n - (n & 1)), visualizeNormals(false) {
	SetName("XGLSphere");
	int i, j;
	float twoPi = (2 * (float)PI);
	XGLVertexAttributes vrtx;
	int halfSeg = nSegments >> 2;
	int halfSegPlus = halfSeg + 1;
	int loopSegments = nSegments;

	for (j = 0; j <loopSegments; j++) {
		for (i = 0; i < halfSegPlus; i++) {
			float angle = (float)i * twoPi / (float)nSegments;
			float angle2 = (float)j * twoPi / (float)nSegments;

			float x = sin(angle)*cos(angle2) * radius;
			float y = cos(angle) * radius;
			float z = (sin(angle)*sin(angle2)) * radius;

			vrtx.t = { (0.5f * x+0.5f), (0.5f * z + 0.5f) };
			vrtx.v = { x, y, z };
			vrtx.c = XGLColors::white;
			vrtx.n = { x / radius, y / radius, z / radius };

			v.push_back(vrtx);
		}
	}

	int nVerts = (int)v.size();

	if (true) {
		j = 0;
		int count = (loopSegments - 2);
		for (j = 0; j < count; j++) {
			for (int i = 0; i < halfSegPlus; i++) {
				idx.push_back((j*halfSegPlus) + i);
				idx.push_back(((j + 1)*halfSegPlus) + i);
			}

			for (int i = 1; i < halfSegPlus; i++) {
				idx.push_back(((j + 2)*halfSegPlus) + halfSegPlus - i);
				idx.push_back(((j + 1)*halfSegPlus) + halfSegPlus - i);
			}
		}
		for (i = 0; i < halfSegPlus; i++) {
			idx.push_back(((j + 1)*halfSegPlus) + i);
			idx.push_back(i);
		}

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
}

void XGLHemiSphere::Draw() {
	glDrawArrays(GL_POINTS, 0, GLsizei(v.size()));
	glDrawElements(GL_TRIANGLE_STRIP, (GLsizei)(idx.size()), XGLIndexType, 0);
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
			vrtx.c = XGLColors::white;
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
			vrtx.c = XGLColors::white;
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
			vrtx.c = { 1, 1, 1, 1 };
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
			XGLVertexAttributes tv = { { v[i].v.x, v[i].v.y, v[i].v.z }, {}, {}, XGLColors::white };
			XGLVertexAttributes tn = { { v[i].n.x, v[i].n.y, v[i].n.z }, {}, {}, XGLColors::white };
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

	v.push_back({ { -t,  t, 0 },{},{}, XGLColors::red });
	v.push_back({ { t, t, 0 }, {}, {}, XGLColors::red });
	v.push_back({ { -t, -t, 0 }, {}, {}, XGLColors::red });
	v.push_back({ { t, -t, 0 }, {}, {}, XGLColors::red });

	v.push_back({ { 0, -t, t }, {}, {}, XGLColors::green });
	v.push_back({ { 0, t, t }, {}, {}, XGLColors::green });
	v.push_back({ { 0, -t, -t }, {}, {}, XGLColors::green });
	v.push_back({ { 0, t, -t }, {}, {}, XGLColors::green });

	v.push_back({ { t, 0, -t }, {}, {}, XGLColors::blue });
	v.push_back({ { t, 0, t }, {}, {}, XGLColors::blue });
	v.push_back({ { -t, 0, -t }, {}, {}, XGLColors::blue });
	v.push_back({ { -t, 0, t }, {}, {}, XGLColors::blue });


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
			vrtx.c = XGLColors::white;
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
	XGLColor white = { 1, 1, 0, 1 };

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

			AddTexture(font.atlasWidth, font.atlasHeight, 1, font.bitmapPages[i]);
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

XGLTexQuad::XGLTexQuad() {
	SetName("XTexQuad");
	const XGLColor white = { 1, 1, 1, 1 };

	v.push_back({ { -1.0, -1.0, 0 }, { 0, 0 }, {}, white });
	v.push_back({ { -1.0, 1.0, 0 }, { 0, 1 }, {}, white });
	v.push_back({ { 1.0, -1.0, 0 }, { 1, 0 }, {}, white });
	v.push_back({ { 1.0, 1.0, 0 }, { 1, 1 }, {}, white });

	idx.push_back(0);
	idx.push_back(1);
	idx.push_back(2);
	idx.push_back(3);
}

XGLTexQuad::XGLTexQuad(int w, int h) {
	SetName("XTexQuad");
	const XGLColor white = { 1, 1, 1, 1 };

	v.push_back({ { 0, 0, 0 }, { 0, 0 }, {}, white });
	v.push_back({ { 0, h, 0 }, { 0, 1 }, {}, white });
	v.push_back({ { w, 0, 0 }, { 1, 0 }, {}, white });
	v.push_back({ { w, h, 0 }, { 1, 1 }, {}, white });

	idx.push_back(0);
	idx.push_back(1);
	idx.push_back(2);
	idx.push_back(3);
}

XGLTexQuad::XGLTexQuad(std::string fileName, int forceChannels) : XGLTexQuad() {
	AddTexture(fileName, forceChannels);
}

XGLTexQuad::XGLTexQuad(int width, int height, int channels, GLubyte *img, bool flipColors) : XGLTexQuad() {
	AddTexture(width, height, channels, img, flipColors);
}
XGLTexQuad::XGLTexQuad(int width, int height, int channels) : XGLTexQuad() {
	AddTexture(width, height, channels);
}

void XGLTexQuad::Reshape(int left, int top, int width, int height) {
	v[0].v.x = (float)left;
	v[0].v.y = (float)top;

	v[1].v.x = (float)left;
	v[1].v.y = (float)height;

	v[2].v.x = (float)width;
	v[2].v.y = (float)top;

	v[3].v.x = (float)width;
	v[3].v.y = (float)height;

	Load(shader, v, idx);
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

// place-holder for geometry-less shapes.  Useful for creating chains
// of "model" matrices.
XGLTransformer::XGLTransformer(){
	SetName("XGLTransformer");
};

XGLSled::XGLSled(bool sa) : showAxes(sa) {
	SetName("XGLSled");

	// 3 lines to represent X,Y,Z axes (orientation)
	// X
	v.push_back({ glm::vec3(0), {}, {}, XGLColors::red });
	v.push_back({ glm::vec3(1.0, 0.0, 0.0) * 5.0f, {}, {}, XGLColors::red });
	// Y
	v.push_back({ glm::vec3(0), {}, {}, XGLColors::green });
	v.push_back({ glm::vec3(0.0, 1.0, 0.0) * 5.0f, {}, {}, XGLColors::green });
	// Z
	v.push_back({ glm::vec3(0), {}, {}, XGLColors::blue });
	v.push_back({ glm::vec3(0.0, 0.0, 1.0) * 5.0f, {}, {}, XGLColors::blue });
}

void XGLSled::Draw() {
	if (showAxes) {
		glDrawArrays(GL_LINES, 0, 6);
		GL_CHECK("glDrawArrays() failed");
	}
}

glm::mat4 XGLSled::GetFinalMatrix() {
	// add the translation of the sled's position for the final model matrix
	return glm::translate(glm::mat4(), p) * glm::toMat4(o);
}

void XGLSled::SampleInput(float yaw, float pitch, float roll) {
	glm::quat rotation;

	// combine yaw,pitch & roll changes into incremental rotation quaternion
	rotation = glm::angleAxis(glm::radians(yaw), glm::vec3(0.0, 0.0, 1.0));
	rotation *= glm::angleAxis(glm::radians(pitch), glm::vec3(1.0, 0.0, 0.0));
	rotation *= glm::angleAxis(glm::radians(roll), glm::vec3(0.0, 1.0, 0.0));

	// Add combined rotationChange to sled's "currentRotation" (orientation) quaternion
	// This order is key to local-relative rotation or world-relative.  This is local-relative
	// Swapping the operand order changes to world-relative order, which is what I had been doing.
	//
	// Can't believe how long it took to figure this out, because it's SO simple now that I know.
	o = o * rotation;

	model = GetFinalMatrix();
}

XGLPointCloud::XGLPointCloud(int nPoints, float radius, XGLColor color, XGLVertex center) : drawFn(nullptr) {
	SetName("XGLPointCloud");

	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> dis(0.0, radius);

	for (int i = 0; i < nPoints; i++) {
		XGLVertex vrtx;
		vrtx.x = dis(gen);
		vrtx.y = dis(gen);
		vrtx.z = dis(gen);
		v.push_back({ vrtx, {}, {}, color });
	}
}

void XGLPointCloud::Draw(){
	if (!drawFn) {
		glDrawArrays(GL_POINTS, 0, GLsizei(v.size()));
		GL_CHECK("glDrawPoints() failed");
	}
	else
		drawFn();
}

XGLCameraFlyer::XGLCameraFlyer(XGL *xgl) : XGLSled(false) {
	// use the left stick to control yaw, right stick to control pitch & roll of the sled (typical R/C transmitter layout)
	// XGLSled::SampleInput(float yaw, float pitch, float roll) also calls XGLSled::GetFinalMatrix()
	xgl->AddProportionalFunc("Xbox360Controller0", [this, xgl](float v) { SampleInput(-v, 0.0f, 0.0f); SetCamera(xgl); });
	xgl->AddProportionalFunc("Xbox360Controller2", [this, xgl](float v) { SampleInput(0.0f, 0.0f, v); SetCamera(xgl); });
	xgl->AddProportionalFunc("Xbox360Controller3", [this, xgl](float v) { SampleInput(0.0f, -v, 0.0f); SetCamera(xgl); });

	// move sled with Xbox360 controller left & right triggers
	xgl->AddProportionalFunc("Xbox360Controller4", [this, xgl](float v) { MoveFunc(xgl, v); });
	xgl->AddProportionalFunc("Xbox360Controller5", [this, xgl](float v) { MoveFunc(xgl, -v); });
}
void XGLCameraFlyer::MoveFunc(XGL* xgl, float v) {
	glm::vec4 f = glm::toMat4(o) * glm::vec4(0.0, v / 10.0f, 0.0, 0.0);
	p += glm::vec3(f);
	SetCamera(xgl);
};

void XGLCameraFlyer::SetCamera(XGL* xgl)
{
	glm::vec3 f = glm::toMat3(o) * glm::vec3(0, 1, 0);
	glm::vec3 u = glm::toMat3(o) * glm::vec3(0, 0, 1);
	xgl->camera.Set(p, f, u);
};

