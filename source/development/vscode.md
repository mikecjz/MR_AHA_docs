# Development and debugging in Visual Studio Code

Set `includePath` in [.vscode/c_cpp_properties.json](https://github.com/mikecjz/MR_AHA/blob/main/.vscode/c_cpp_properties.json) to include
`include/`, `src/common/`, `src/cuda/`, `third_party/finufft/include/` and the CUDA and MATLAB headers for IntelliSense.

To debug the MEX gateways: build with `-DCMAKE_BUILD_TYPE=Debug`, point `program` in
[.vscode/launch.json](https://github.com/mikecjz/MR_AHA/blob/main/.vscode/launch.json) at your MATLAB executable, and set breakpoints in
`matlab/*.cu` or `src/cuda/*.cu`. The [MATLAB GPU CUDA and MEX Programming
Manual](https://www.mathworks.com/help/parallel-computing/gpu-cuda-and-mex-programming.html) is the
reference for the MathWorks side.
