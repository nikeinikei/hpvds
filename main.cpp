#include <memory>
#include <iostream>

import Graphics;
import Timer;


int main() {
    try {
        std::unique_ptr<Graphics> graphics = std::make_unique<Graphics>();

        std::vector<std::unique_ptr<Model>> models;

        models.push_back(std::move(graphics->createModel("resources/cube.glb")));
        models[0]->color = Color3(0.1647058823529412f, 0.3137254901960784f, 0.7450980392156863f);

        while (!graphics->shouldClose()) {
            graphics->pollEvents();
            graphics->render(models);
        }

        graphics->waitIdle();

        models.clear();
    }
    catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
