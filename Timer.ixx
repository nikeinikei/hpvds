module;

#include <chrono>
#include <iostream>

export module Timer;

export class Timer {
public:
	Timer() {
		last = lastFrame = start = std::chrono::high_resolution_clock::now();
	}

	float tick() {
		frames++;
		auto current = std::chrono::high_resolution_clock::now();

		float dt = std::chrono::duration<float, std::chrono::seconds::period>(current - last).count();

		auto frameDuration = current - lastFrame;
		if (frameDuration.count() >= 1000000000LL) {
			lastFrame = current;
			std::cout << "fps: " << frames << std::endl;
			frames = 0;
		}

		last = current;
		return dt;
	}

	float elapsed() {
		auto current = std::chrono::high_resolution_clock::now();
		return std::chrono::duration<float, std::chrono::seconds::period>(current - start).count();
	}

private:
	std::chrono::steady_clock::time_point start;
	std::chrono::steady_clock::time_point lastFrame;
	std::chrono::steady_clock::time_point last;
	float lastTime = 0.0f;
	unsigned frames = 0;

};
