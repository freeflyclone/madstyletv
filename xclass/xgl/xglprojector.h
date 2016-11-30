// modify the "perspective" matrix according to window shape

class XGLProjector {
public:
	typedef std::function<void(int, int)> ReshapeFunc;
	typedef std::vector<ReshapeFunc> ReshapeCallbackList;

	XGLProjector() : width(1), height(1) {};
    void Reshape(int w, int h);
	void Reshape();
	glm::mat4 GetProjectionMatrix();
	glm::mat4 GetOrthoMatrix();
	void AddReshapeCallback(ReshapeFunc);

	int width, height;
	ReshapeCallbackList callbacks;
};

