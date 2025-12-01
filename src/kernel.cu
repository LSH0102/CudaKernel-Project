#include <cuda.h>
#include <cstddef>
#include <stdio.h>

///用于测试用的向量加法内核
__global__ void vector_add_kernel(float* a, float* b,float*c,const int size)
{
	int id=blockIdx.x*blockDim.x+threadIdx.x;
	if(id<size)
	{
		c[id]=a[id]+b[id];
	}
	return;
}

///调用内核的主程序
extern "C"
{
	void launch_add(float* a,float* b,float *c, const int size, cudaStream_t stream)
	{

		int input_size=sizeof(float)*size;
		int output_size=input_size;
		float * d_a, *d_b,*d_c;
		cudaMalloc((void**)&d_a,input_size);
		cudaMalloc((void**)&d_b,input_size);
		cudaMalloc((void**)&d_c,output_size);

		cudaMemcpy(d_a,a,input_size,cudaMemcpyHostToDevice);
		cudaMemcpy(d_b,b,input_size,cudaMemcpyHostToDevice);
		cudaMemcpy(d_c,c,output_size,cudaMemcpyHostToDevice);

		int nthread=256;
		int ngrid=(size+nthread-1)/nthread;
		dim3 grid_dim=(ngrid);
		dim3 block_dim=(nthread);

		vector_add_kernel<<<grid_dim,block_dim,0,stream>>>(d_a,d_b,d_c,size);
		cudaMemcpy(c,d_c,output_size,cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);


	}
}