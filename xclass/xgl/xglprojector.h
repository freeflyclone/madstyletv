// modify the "perspective" matrix according to window shape

class XGLProjector {
public:
	typedef std::function<void(int, int)> ReshapeFunc;

	XGLProjector() : width(1), height(1) {};
    void Reshape(int w, int h);
	void Reshape();
	glm::mat4 GetProjectionMatrix();
	void AddReshapeCallback(ReshapeFunc);

	int width, height;
	ReshapeFunc reshapeCallback;
};

