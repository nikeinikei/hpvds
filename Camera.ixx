module;

#include "glm_config.h"
#include <GLFW/glfw3.h>

export module Camera;

constexpr float cameraSpeed = 5.0f;

export class Camera {
public:
	Camera(GLFWwindow* window, float fov)
		: window(window), position(0.0f), fov(fov), pitch(0.0f), yaw(0.0f) {
		position = glm::vec3(0.0f, 0.0f, 2.0f);
		yaw = glm::radians(30.0f);
	}

	void update(float dt) {
		glm::vec4 direction(0.0f, 0.0f, 0.0f, 1.0f);

		if (glfwGetKey(window, GLFW_KEY_W)) {
			direction.z -= 1.0f;
		}
		if (glfwGetKey(window, GLFW_KEY_A)) {
			direction.x -= 1.0f;
		}
		if (glfwGetKey(window, GLFW_KEY_S)) {
			direction.z += 1.0f;
		}
		if (glfwGetKey(window, GLFW_KEY_D)) {
			direction.x += 1.0f;
		}

		if (direction.x != 0.0f || direction.z != 0.0f) {
			position += dt * cameraSpeed * glm::vec3(glm::rotate(glm::mat4(1.0f), yaw, glm::vec3(0.0f, -1.0f, 0.0f)) * glm::normalize(direction));
		}
	}

	glm::mat4 getProjection() {
		return glm::scale(glm::perspective(glm::radians(90.0f), fov, 0.1f, 100.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	}

	glm::mat4 getView() {
		return glm::translate(glm::mat4(1.0f), -position);
	}

private:
	GLFWwindow* window;
	float fov;
	glm::vec3 position;
	float pitch;
	float yaw;
};
