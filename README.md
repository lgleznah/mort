# mort
My Own RayTracer - A personal project of writing a raytracer in CUDA.

After having read through the amazing [Ray Tracing in One Weekend](https://raytracing.github.io/) book series, and also after having read about CUDA, I decided to try porting the raytracer to work on the GPU.

# How to run
Go to [the latest release of the repository](https://github.com/lgleznah/mort/releases), and download both `mort.exe` and `glut64.dll`.

Go to where the files have been downloaded, open a terminal there, and write: `mort.exe <number_of_scene_between_1_and_10>`

For reference, Scene 1 takes around 16 seconds to render on a Nvidia GeForce RTX 2080.

# How to build
This code has been built on a Windows 10 machine with both CUDA 11.3 (compiled on Visual Studio 2019) and with CUDA 13.2 (compiled on Visual Studio 2022).

This code has been tested in two devices with the following specs:

- Device 1 specs:
  Device Name	DESKTOP-BBS5DPR
  Processor	Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz   2.59 GHz
  Installed RAM	32,0 GB (31,8 GB usable)
  Storage	954 GB SSD SAMSUNG MZVLB1T0HALR-00000
  Graphics Card	NVIDIA GeForce RTX 2080 (8 GB), Intel(R) UHD Graphics 630 (128 MB)
  Device ID	B580CE2A-2D6A-4F2F-B22A-3D67CF8B64CF
  Product ID	00326-30000-00001-AA971
  System Type	64-bit operating system, x64-based processor
  Pen and touch	No pen or touch input is available for this display

- Device 1 Windows specs:
  Edition	Windows 10 Home
  Versión	22H2
  Installed on	‎01/‎10/‎2025
  OS Build	19045.6466

- Device 2 specs:
  Device Name	DESKTOP-L6OFD1B
  Processor	13th Gen Intel(R) Core(TM) i9-13900KF   3.00 GHz
  Installed RAM	64.0 GB (63.8 GB usable)
  Storage	3.64 TB HDD ST4000DM004-2CV104, 1.82 TB SSD WD_BLACK SN770 2TB
  Graphics Card	NVIDIA GeForce RTX 4070 (12 GB)
  Device ID	C5040BDA-8DC8-4548-8E5E-65D90A2549C1
  Product ID	00331-60014-47712-AA760
  System Type	64-bit operating system, x64-based processor
  Pen and touch	No pen or touch input is available for this display

- Device 2 Windows specs:
  Edition	Windows 10 Pro N
  Version	22H2
  Installed on	‎19-‎Dec-‎23
  OS Build	19045.6466

Make sure to install the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) (select your OS, architecture, OS version, local installer, download and execute the CUDA Toolkit installer).

Open `mort.sln` on Visual Studio, and compile the project on Release mode. CUDA might have to be added as a build customization (right click on the project->Build dependencies->Build customizations, and select the CUDA version matching the one you installed before).

In case the project does not compile, it is likely that the `mort.cu` file appears as not added to the project, and once added, its file type is "Does not participate in build". In this case, after adding `mort.cu` to the project, change its file type to CUDA C/C++ (right click on the `mort.cu` file->Properties->General, and change the element type to CUDA C/C++).

After compiling, the resulting executable will be at `bin\win64\Release\mort.exe`. To run it, in the root of the repository, run `bin\win64\Release\mort.exe <number_of_scene_between_1_and_10>`.
