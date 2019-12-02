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

		const aiScene* scene;
		scene = importer.ReadFile
		(
			fileName,
			aiProcess_Triangulate | aiProcess_SortByPType
		);

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
			ConvertMeshToXglVertexList(mesh);
			ConvertMeshToXglFaces(mesh);
		}
	}

	void ConvertMeshToXglVertexList(aiMesh* mesh)
	{
		for (int index = 0; index < mesh->mNumVertices; index++)
		{
			aiVector3D vrtx = mesh->mVertices[index];
			aiVector3D n = mesh->mNormals[index];

			// normals from model *might* not be unit length
			// according to Assimp docs.
			XGLVertex nrml = glm::normalize(glm::vec3(n.x,n.y,n.z));

			v.push_back
			(
				{ 
					{ vrtx.x, vrtx.y, vrtx.z }, 
					{}, 
					nrml,
					XGLColors::yellow 
				}
			);
		}
	}

	void ConvertMeshToXglFaces(aiMesh* mesh)
	{
		for (int index = 0; index < mesh->mNumFaces; index++)
		{
			aiFace face = mesh->mFaces[index];

			for (int j = 0; j < face.mNumIndices; j++)
				idx.push_back(face.mIndices[j]);
		}
	}

	void Draw()
	{
		glDrawElements(GL_TRIANGLES, (GLsizei)idx.size(), XGLIndexType, 0);
		GL_CHECK("glDrawElements() failed");
	}
};

void ExampleXGL::BuildScene() 
{
	XGLAssimp *shape;

	std::string shapePath = pathToAssets + "/assets/3DModels/mm5.stl";

	AddShape
	(
		"shaders/specular", 
		[&]() {	shape = new XGLAssimp(shapePath); return shape;	}
	);

	shape->attributes.diffuseColor = XGLColors::yellow;
}
