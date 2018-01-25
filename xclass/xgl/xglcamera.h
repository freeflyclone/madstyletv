// modify the "view" matrix to move the Camera(s)
class XGLCamera : public XObject {
public:
    typedef std::function<void(XGLCamera *)> XGLCameraFunk;

    XGLCamera();

	void Set(glm::vec3 pos, glm::vec3 front, glm::vec3 up);

    void SetTheFunk(XGLCameraFunk);
    void Animate();

	glm::mat4 GetViewMatrix();

 	glm::vec3 pos;
	glm::vec3 front;
	glm::vec3 up;

private:
    XGLCameraFunk funk;
};
