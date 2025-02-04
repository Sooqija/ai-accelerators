# Build Instructions (Windows)

Install the `nvcc` compiler. It is used for compiling CUDA code and requires the `cl` compiler to build C++ host code.
Ensure that the path to `cl` is included in your system's `PATH` environment variable. If not, check the common location: `C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/<version>/bin/HostX64/x64`

Create a directory for the build:
```bash
mkdir build
```

To compile the `main.cu` file, run the following command:
```bash
nvcc main.cu -o ./build/main.exe
```

To pass flags to the `cl` compiler, use the `-Xcompiler` flag. For example, to enable **OpenMP**, run:
```bash
nvcc main.cu -o ./build/main.exe -Xcompiler /openmp:experimental
```

Here are some other useful flags you can use with `nvcc`:
- `/Qvec-report:1` — generates a vectorization report.
- `/fp:fast` — helps the compiler vectorize expressions within loops.

# Launch Instructions

Softmax parameters:
+ `n` - size of matrices

Convolution parameters:
+ `n` - size of input matrices
+ `k` - size of filter matrix
+ `m` - batch size

Examples.

```bash
./softmax.exe 1024
```

```bash
./conv.exe 1024 3 16
```

# VScode Script

```shell
@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
code .
exit
```