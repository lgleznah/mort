# mort
A personal project of writing a raytracer in CUDA.

After having read through the amazing [Ray Tracing in One Weekend](https://raytracing.github.io/) book series, and also after having read about CUDA, I decided to try porting the raytracer to work on the GPU.

# How to run
In the root of the repository, open a terminal and write the following command:

`bin\win64\Release\mort.exe <number_of_scene_between_1_and_10>`

For reference, Scene 1 takes around 16 seconds to render on a Nvidia GeForce RTX 2080

# How to build
This code has been built on a Windows 10 machine. It has both been tested with CUDA 11.3 (compiled on Visual Studio 2019) and with CUDA 13.2 (compiled on Visual Studio 2022).

Open `mort.sln` on Visual Studio, and compile the project. CUDA might have to be added as a build customization (right click on the project->Build dependencies->Build customizations, and select CUDA), and `mort.cu` file type might have to be set as CUDA C/C++ (right click on the `mort.cu` file->Properties->General, and change the element type to CUDA C/C++).
