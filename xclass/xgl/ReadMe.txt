XGL Notes: (in no particular order)
-----------------------------------

The term "shape" is used to generically describe a 3D world object.
The term was chosen to disambiguate "object" which has meaning
in a C++ context.

A shape consists of various items including geometry and rendering
attributes consisting of vertex attributes and uniform variables, and
the requisite GLSL shader program for the rendering pipeline.  Vertex
attributes describe "what" to rasterize geometrically, the uniforms 
describe "how" to rasterize by allowing to specify parameters to
the rasterization process.

In OpenGL, changing GLSL shader programs is somewhat expensive in terms
of real-time rendering performance, so the XGL framework organizes
shapes according to the shader(s) they use, and renders in a 
sequential "shapes-per-shader" order.

Uniform variable management:
----------------------------
Uniforms are variables allowing CPU <-> GPU code communication.

A Uniform Buffer Object (UBO) can be used to share variables shaders
for things that are of "scene scope" vs "shape scope". In other
words: global to all shaders.

Uniform blocks can contain multiple variables, basically the 
equivalent of a C/C++ struct.

A uniform block MUST be in a uniform buffer object.(?)

A "member" variable of a uniform block object can be referenced
from OpenGL code as if it were a regular uniform.  Ie: by getting
it's location with glGetUniformLoation()/glProgramUniform(), but
only if it's in the currently bound uniform block index.(?)


