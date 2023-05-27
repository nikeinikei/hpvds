export module Color;

export struct Color3 {
public:
	Color3(): r(1.0f), g(1.0f), b(1.0f) { }
	Color3(float r, float g, float b) : r(r), g(g), b(b) {}

	float r;
	float g;
	float b;
};
