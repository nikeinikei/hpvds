module;

#include "glm_config.h"
#include <GLFW/glfw3.h>

export module Camera;

constexpr float cameraSpeed = 5.0f;
constexpr float rotationSpeed = 2.5f;

export class Camera {
public:
	Camera(GLFWwindow* window, float fov)
		: window(window), position(0.0f), fov(fov), pitch(0.0f), yaw(0.0f) {
		position = glm::vec3(0.0f, 0.0f, 2.0f);
		yaw = glm::radians(10.0f);
	}

	void update(float dt) {
        float sin = std::sin(yaw) * cameraSpeed * dt;
        float cos = std::cos(yaw) * cameraSpeed * dt;

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_TRUE) {
            position.z -= cos;
            position.x += sin;
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_TRUE) {
            position.z += cos;
            position.x -= sin;
        }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_TRUE) {
            position.z -= sin;
            position.x -= cos;
        }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_TRUE) {
            position.z += sin;
            position.x += cos;
        }
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_TRUE) {
            position.y += cameraSpeed * dt;
        }
        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_TRUE) {
            position.y -= cameraSpeed * dt;
        }

        if (glfwGetKey(window, GLFW_KEY_I) == GLFW_TRUE) {
            pitch -= rotationSpeed * dt;
        }
        if (glfwGetKey(window, GLFW_KEY_K) == GLFW_TRUE) {
            pitch += rotationSpeed * dt;
        }
        if (glfwGetKey(window, GLFW_KEY_J) == GLFW_TRUE) {
            yaw -= rotationSpeed * dt;
        }
        if (glfwGetKey(window, GLFW_KEY_L) == GLFW_TRUE) {
            yaw += rotationSpeed * dt;
        }
	}

	glm::mat4 getProjection() {
		return glm::scale(glm::perspective(glm::radians(90.0f), fov, 0.1f, 100.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	}

	glm::mat4 getView() {
        glm::mat4 view(1.0f);
        view = glm::rotate(view, pitch, glm::vec3(1.0f, 0.0f, 0.0f));
        view = glm::rotate(view, yaw, glm::vec3(0.0f, 1.0f, 0.0f));
        view = glm::translate(view, -position);
        return view;
	}

private:
	GLFWwindow* window;
	float fov;
	glm::vec3 position;
	float pitch;
	float yaw;
};
