#include "ExampleXGL.h"

class XGLGraph : public XGLShape {
public:
	XGLGraph();

    void Draw();

	void NewValue(float);

private:
	std::mutex mutex;
	std::vector<float> values;
	int nValues;
	int currentOffset;
};
