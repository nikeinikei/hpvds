#include <memory>
#include <iostream>

import Graphics;


int main() {
    try {
        std::unique_ptr<Graphics> graphics = std::make_unique<Graphics>();
        graphics->runMainLoop();
    }
    catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
