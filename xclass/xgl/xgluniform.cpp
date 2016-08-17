#include "xgl.h"

XGLUniformf::XGLUniformf(GLint program, std::string n, glm::vec3 v) : name(n){
    location = glGetUniformLocation(program, name.c_str());
    GL_CHECK("getUniformLocation failed");
    if (location == -1)
        return;
    glGetUniformfv(program, location, glm::value_ptr(v));
    GL_CHECK("glGetUniformfv() failed");
}
