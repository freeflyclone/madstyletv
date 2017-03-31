#include "ExampleXGL.h"

class XGLGraph : public XGLShape {
public:
	XGLGraph();

    void Draw();

private:
	std::vector<float> values;
	int nValues;
	int currentOffset;
};
