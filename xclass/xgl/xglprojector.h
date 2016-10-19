// modify the "perspective" matrix according to window shape

class XGLProjector {
public:
    XGLProjector() : width(1), height(1) {};
    void Reshape(int w, int h);
	void Reshape();
	glm::mat4 GetProjectionMatrix();

	int width, height;
};

