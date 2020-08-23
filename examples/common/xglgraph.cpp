#include "xglgraph.h"

XGLGraph::XGLGraph() {
	SetName("XGLGraph");

	int numSamples = 1000;
	float xStart = -20.0f;
	float xStep =(float)(xStart*-2) / (float)numSamples;

	for (int i=0; i<numSamples; i++) {
		float y = sin((float)i / 10.0f);
		values.push_back(y);
	}

	for (int i=0; i<values.size(); i++) {
		float x = xStart + (xStep*(float)i);
		v.push_back({{x,values[i],0.05f},{},{0,0,1}, XGLColors::green});
	}

	for (int i=0; i<values.size()-1; i++) {
		idx.push_back(i);
		idx.push_back(i+1);
	}

	currentOffset = 0;
}

XGLGraph::XGLGraph(std::vector<float>input) {
	SetName("XGLGraph");

	int numSamples = input.size();
	float xStart = -20.0f;
	float xStep = (float)(xStart*-2) / (float)numSamples;

	if (values.size())
		values.clear();

	values = input;

	for (int i = 0; i < values.size(); i++) {
		float x = xStart + (xStep*(float)i);
		v.push_back({ {x,values[i],0.05f},{},{0,0,1}, XGLColors::green });
	}

	for (int i = 0; i < values.size() - 1; i++) {
		idx.push_back(i);
		idx.push_back(i + 1);
	}

	currentOffset = 0;
}

void XGLGraph::Draw() {
	glDrawElements(GL_LINES, (GLsizei)(idx.size()), XGLIndexType, 0);
	GL_CHECK("glDrawElements() failed");
    return;
}

void XGLGraph::NewValue(float value) {
	std::unique_lock<std::mutex> lock(mutex);

	values[currentOffset] = value;
	currentOffset = (currentOffset + 1) % values.size();
}
