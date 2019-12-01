/**************************************************************
** AssimpDevBuildScene.cpp
**
** Demonstrate the Assimp library.
**************************************************************/
#include "ExampleXGL.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

class XGLAssimp : public XGLShape
{
public:
	XGLAssimp(const std::string fileName)
	{
		Assimp::Importer importer;

		const aiScene* scene = importer.ReadFile(fileName, 0);
		if (scene == nullptr)
		{
			xprintf("Failed to load file '%s'\n", fileName.c_str());
			return;
		}

		ConvertScene(scene);
	}

	void ConvertScene(const aiScene* scene)
	{
		for (int meshIndex = 0; meshIndex < scene->mNumMeshes; meshIndex++)
		{
			aiMesh* mesh = scene->mMeshes[meshIndex];
			ConvertMesh(mesh);
		}
	}

	void ConvertMesh(aiMesh* mesh)
	{
		for (int index = 0; index < mesh->mNumVertices; index++)
		{
			aiVector3D vrtx = mesh->mVertices[index];

			v.push_back
			(
				{ 
					{ vrtx.x, vrtx.y, vrtx.z }, 
					{}, 
					{}, 
					XGLColors::yellow 
				}
			);
		}
	}

	void Draw()
	{
		glPointSize(4.0f);
		glDrawArrays(GL_POINTS, 0, GLsizei(v.size()));
		GL_CHECK("glDrawArrays() failed");
	}
};

void ExampleXGL::BuildScene() 
{
	XGLAssimp *shape;

	std::string shapePath = pathToAssets + "/assets/3DModels/motor-mount-3.stl";

	AddShape
	(
		"shaders/000-simple", 
		[&]() 
		{ 
			shape = new XGLAssimp(shapePath); return shape; 
		}
	);
}
