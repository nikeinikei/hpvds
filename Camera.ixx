module;

#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

export module Camera;

export class Camera {
public:
	Camera()
		: position(0.0f) {
	}

	void setPosition(glm::vec3 pos) {
		position = pos;
	}

	glm::mat4 getProjection() {
		return glm::mat4(1.0f);
		// return glm::perspective(90.0f, 1.0f, 0.1f, 1000.0f);
	}

	glm::mat4 getView() {
		return glm::rotate(glm::mat4(1.0f), 3.14f, glm::vec3(0.0f, 0.0f, 1.0f));
	}

private:
	glm::vec3 position;

};
