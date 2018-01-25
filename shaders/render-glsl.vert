//----------------------------------------------------------------------------------
// File:        es3aep-kepler\ComputeParticles\assets\shaders/renderVS.glsl
// SDK Version: v2.11 
// Email:       gameworks@nvidia.com
// Site:        http://developer.nvidia.com/
//
// Copyright (c) 2014-2015, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//----------------------------------------------------------------------------------
// Version ID added via C-code prefixing
#version 430
#extension GL_EXT_shader_io_blocks : enable

precision highp float;
#ifndef __cplusplus

#define sdk_bool bool
// Standard SDK defines
#define SDK_BOOL  bool
#define SDK_VEC2  vec2
#define SDK_VEC3  vec3
#define SDK_VEC4  vec4
#define SDK_MAT4  mat4

layout(std140, binding=1) uniform
#else
struct
#endif
    
ShaderParams
{
    SDK_MAT4 ModelView;
    SDK_MAT4 ModelViewProjection;
    SDK_MAT4 ProjectionMatrix;

    SDK_VEC4 attractor;

    uint  numParticles;
    float spriteSize;
    float damping;

    float noiseFreq;
    float noiseStrength;

#ifdef CPP
    ShaderParams() :
        spriteSize(0.015f),
        attractor(0.0f, 0.0f, 0.0f, 0.0f),
        damping(0.95f),
        noiseFreq(10.0f),
        noiseStrength(0.001f)
        {}
#endif
};

#define WORK_GROUP_SIZE 128

layout( std140, binding=1 ) buffer Pos {
    vec4 pos[];
};

out gl_PerVertex {
    vec4 gl_Position;
};

out block {
     vec4 color;
     vec2 texCoord;
} Out;

void main() {
    // expand points to quads without using GS
    int particleID = gl_VertexID >> 2; // 4 vertices per particle
    vec4 particlePos = pos[particleID];

    Out.color = vec4(0.5, 0.2, 0.1, 1.0);

    //map vertex ID to quad vertex
    vec2 quadPos = vec2( ((gl_VertexID - 1) & 2) >> 1, (gl_VertexID & 2) >> 1);

    vec4 particlePosEye = ModelView * particlePos;
    vec4 vertexPosEye = particlePosEye + vec4((quadPos*2.0-1.0)*spriteSize, 0, 0);

    Out.texCoord = quadPos;
    gl_Position = ProjectionMatrix * vertexPosEye;
}
