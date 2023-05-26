@echo off
glslc shader.vert -o vert.spv
glslc shader.frag -o frag.spv
xxd -i vert.spv vertexShader.h
xxd -i frag.spv fragmentShader.h
