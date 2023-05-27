module;

#include "glm_config.h"

export module Camera;

export class Camera {
public:
	Camera(float fov)
		: position(0.0f), fov(fov) {
	}

	void setPosition(glm::vec3 pos) {
		position = pos;
	}

	glm::mat4 getProjection() {
		return glm::scale(glm::perspective(glm::radians(90.0f), fov, 0.1f, 100.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	}

	glm::mat4 getView() {
		return glm::lookAt(glm::vec3(2.0f, -2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	}

private:
	float fov;
	glm::vec3 position;

};
