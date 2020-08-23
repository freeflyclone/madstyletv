#include "ExampleXGL.h"

class XGLGraph : public XGLShape {
public:
	XGLGraph();
	XGLGraph(std::vector<float>);

    void Draw();

	void NewValue(float);

private:
	std::mutex mutex;
	std::vector<float> values;
	int currentOffset{ 0 };
};
