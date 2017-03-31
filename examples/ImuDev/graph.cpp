#include "graph.h"

XGLGraph::XGLGraph() {
	SetName("XGLGraph");

	int numSamples = 1000;
	float xStart = -10.0f;
	float xStep = 20.0f / 1000.0f;

	for (int i=0; i<numSamples; i++) {
		float y = sin((float)i / 10.0f);
		values.push_back(y);
	}

	for (int i=0; i<24; i++) {
		values.push_back(0.0f);
	}

	for (int i=0; i<values.size(); i++) {
		float x = xStart + (xStep*(float)i);
		v.push_back({{x,values[i],5.0f},{},{}, XGLColors::green});
	}

	for (int i=0; i<values.size()-1; i++) {
		idx.push_back(i);
		idx.push_back(i+1);
	}

	nValues = values.size();
	currentOffset = 0;
}

void XGLGraph::Draw() {
	int i,j;

	XGLVertexAttributes *vb = MapVertexBuffer();

	// cycle throught the "values" buffer, oscilloscope style...
	for (i=currentOffset, j=0; i<nValues; i++,j++)
		vb[j].v.y = values[i];

	for (i=0; i<currentOffset; i++,j++)
		vb[j].v.y = values[i];

	UnmapVertexBuffer();

	currentOffset = (currentOffset + 1) % nValues;

	glDrawElements(GL_LINES, (GLsizei)(idx.size()), XGLIndexType, 0);
	GL_CHECK("glDrawElements() failed");
    return;
}
