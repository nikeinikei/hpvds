module;

#include <chrono>
#include <iostream>

export module Timer;

export class Timer {
public:
	Timer() {
		last = start = std::chrono::high_resolution_clock::now();
	}

	void tick() {
		frames++;
		auto current = std::chrono::high_resolution_clock::now();
		auto duration = current - last;
		if (duration.count() >= 1000000000LL) {
			last = current;
			std::cout << "fps: " << frames << std::endl;
			frames = 0;
		}
	}

	float elapsed() {
		auto current = std::chrono::high_resolution_clock::now();
		auto duration = current - start;
		return static_cast<float>(duration.count()) / 1e9f;
	}

private:
	std::chrono::steady_clock::time_point start;
	std::chrono::steady_clock::time_point last;
	unsigned frames = 0;

};
