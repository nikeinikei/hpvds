module;

#include "glm_config.h"

#include <memory>

export module Model;

import Buffer;

export class Model {
public:
    Model(std::unique_ptr<Buffer>& vertexBuff, std::unique_ptr<Buffer>& indexBuff, size_t numVertices)
        : numVertices(numVertices), modelTransformation(1.0f) {
        vertexBuffer = std::move(vertexBuff);
        indexBuffer = std::move(indexBuff);
    }

    std::unique_ptr<Buffer> vertexBuffer;
    std::unique_ptr<Buffer> indexBuffer;
    size_t numVertices;
    glm::mat4 modelTransformation;
};
