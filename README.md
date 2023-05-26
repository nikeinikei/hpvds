# Build

Currently only Windows is supported.

There are a few prerequisites that need to be installed for the build to work.

- [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows)
- A few dependencies from [vcpk](https://github.com/microsoft/vcpkg):
```bash
vcpkg install glfw3:x64-windows
vcpkg install glm:x64-windows
vcpkg integrate install
```

If these prerequisites have been installed, then building should be as easy as opening the hpvds.sln file
and compiling from within visual studio.

If the shaders need to be recompiled, `xxd` needs to be available.
The easiest way is to install [Vim](https://www.vim.org/) and then adding its installation location to the path.
Now it should be as easy as:

```bash
cd shaders
compile.bat
```
