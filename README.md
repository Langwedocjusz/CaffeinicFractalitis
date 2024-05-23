# CaffeinicFractalitis

Small command line program for drawing fractals, created with primary aim of learning about compiler intrinsics and secondary aim of learning a bit about fractal drawing techniques. Implemented routines use both SIMD and multithreading for maximum performance.

## Example Images

| Smooth iteration count | Gradient |
| ------------ | ------------- |
| ![alt text](https://github.com/Langwedocjusz/CaffeinicFractalitis/blob/main/img/SmoothIter.png?raw=true) | ![alt text](https://github.com/Langwedocjusz/CaffeinicFractalitis/blob/main/img/Gradient.png?raw=true) | 

## Building 
Regardless of the platform make sure cmake is installed and added to your path.

### On Linux:
For building you can use the provided script `BuildProjects.sh`. It will ask you to choose either 'Debug' or 'Release' configuration. The executable will be created in the `build/bin` directory.
  
### On Windows:
The provided batchfile `WIN_GenerateProjects.bat` will generate a Visual Studio solution.
After running it you can open `build/LofiLandscapes.sln` to select configuration and build the program.

## Running
The only required argument of the program is a path to a config json file. For reference you can use the one provided with the repo:

	./build/bin/CaffeinicFractalitis example.json

Additionaly you can set execution flags, specifying number of threads and simd instruction type:

	./build/bin/CaffeinicFractalitis example.json -j <NUM THREADS> <SIMD FLAG>

Available simd flags are `-Scalar`, `-SSE` and `-AVX`

