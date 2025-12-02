#include <cuda.h>
#include <cstddef>
#include <stdio.h>

const int TILE_SIZE=128;
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

/// 批量矩阵乘法内核
__global__ void bmm_kernel(float *a,float* b,float *c,const int batch_size, const int m, const int n, const int p)
{
	//每个block处理一对矩阵
	int batch_id=blockIdx.x;

	//每个thread处理一个 TILE_SIZE x TILE_SIZE 的小矩阵块 i.e左上角为 TILE_SIZE *i,TILE_SIZE *j 的矩阵块
	int i=threadIdx.x;
	int j=threadIdx.y;
	int k=0;
	int row=0;
	int column=0;
	float temp=0.0;

	//计算c中第row行, column列的结果
	for(row=i*TILE_SIZE;row<m&& row<(i+1)*TILE_SIZE;row++)
	{
		for(column=j*TILE_SIZE;column<p&&column<(j+1)*TILE_SIZE;column++)
		{
			temp=0.0;
			for(k=0;k<n;k++)
			{
				temp+=a[batch_id*m*n+n*row+k]*b[batch_id*n*p+p*k+column];
			}
			c[batch_id*m*p+p*row+column]=temp;
		}
	}
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

	void launch_bmm(float *a,float* b,float *c,const int batch_size, const int m, const int n, const int p,cudaStream_t stream )
	{
		int a_size=sizeof(float)*m*n*batch_size;
		int b_size=sizeof(float)*p*n*batch_size;
		int c_size=sizeof(float)*m*p*batch_size;

		float * d_a, *d_b,*d_c;
		cudaMalloc((void**)&d_a,a_size);
		cudaMalloc((void**)&d_b,b_size);
		cudaMalloc((void**)&d_c,c_size);

		cudaMemcpy(d_a,a,a_size,cudaMemcpyHostToDevice);
		cudaMemcpy(d_b,b,b_size,cudaMemcpyHostToDevice);
		cudaMemcpy(d_c,c,c_size,cudaMemcpyHostToDevice);

		//支持的最大长与宽为32*TILE_SIZE
		int nthreadx=(m+TILE_SIZE-1)/TILE_SIZE;
		int nthready=(p+TILE_SIZE-1)/TILE_SIZE;
		
		dim3 grid_dim(batch_size);
		dim3 block_dim(nthreadx,nthready);

		bmm_kernel<<<grid_dim,block_dim,0,stream>>>(d_a,d_b,d_c, batch_size,m,n,p);

		cudaMemcpy(c,d_c,c_size,cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);
	}
}