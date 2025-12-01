### My project of writing cuda kernels
Used only for study and communication purpose

To compile kernels, use the following command

`nvcc -Xcompiler -fPIC -shared src/kernel.cu -o cuda_kernels/kernel.so`

then 

`python test.py`

## Functions
1. Vector addition (added on 2025/12/1)
