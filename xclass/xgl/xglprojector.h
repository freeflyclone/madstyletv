// modify the "perspective" matrix according to window shape

class XGLProjector {
public:
    void Reshape(int w, int h);
	glm::mat4 GetProjectionMatrix();

private:
	int width, height;
};

