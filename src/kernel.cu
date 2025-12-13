#include <cuda.h>
#include <cstddef>
#include <stdio.h>

const int TILE_SIZE=64;
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

///shift函数内核, 用于读取 X[:,i:i+width*stride:stride,j:j+height*stride:stride,:]
__global__ void shift_kernel(float* x, float* res, int i,int j,int stride, const int input_H, const int input_W, const int out_H, const int out_W,const int in_channel)
{
	//每个block处理一个矩阵
	int batch_id=blockIdx.x;
	//每个thread处理一张图片的若干部分
	
	int segment_x=(out_H+32)/32;
	int segment_y=(out_W+32)/32;
	int m,n,p;
	for(m=0;m<segment_x&&threadIdx.x*segment_x+m<out_H;m++)
	{
		for(n=0;n<segment_y&&threadIdx.y*segment_y+n<out_W;n++)
		{
			//准备处理 (threadIdx.x*segment_x+m, threadIdx.y*segment_y+n)的数据
			for(p=0;p<in_channel;p++)
			{
				res[batch_id*out_H*out_W*in_channel+(threadIdx.x*segment_x+m)*out_W*in_channel+(threadIdx.y*segment_y+n)*in_channel+p]=x[batch_id*input_H*input_W*in_channel+(i+stride*(threadIdx.x*segment_x+m))*input_W*in_channel+(j+stride*(threadIdx.y*segment_y+n))*in_channel+p];
			}
		}
	}



}

///conv2d的特殊bmm 此时可以把矩阵b看成每个batch都相同的矩阵, 即batch_id永远等于0
__global__ void conv_bmm(float *shift, float* filter, float *c,int batch_size,const int m,const int n, const int p)
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
				temp+=shift[batch_id*m*n+n*row+k]*filter[p*k+column];
			}
			c[batch_id*m*p+p*row+column]+=temp;
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

	void launch_shift(float* x, float* res, int i,int j,int batch_size,int stride,const int input_H, 
							const int input_W, const int out_H, const int out_W,const int in_channel,cudaStream_t stream)
	{
		int input_size=sizeof(float)*input_H*input_W*in_channel*batch_size;
		int out_size=sizeof(float)*out_H*out_W*in_channel*batch_size;

		float *d_in,*d_res;
		cudaMalloc((void**)&d_in,input_size);
		cudaMalloc((void**)&d_res,out_size);

		cudaMemcpy(d_in,x,input_size,cudaMemcpyHostToDevice);

		dim3 grid_dim(batch_size);
		dim3 block_dim(32,32,1);

		shift_kernel<<<grid_dim,block_dim,0,stream>>>(d_in,d_res,i,j,stride,input_H,input_W,out_H,out_W,in_channel);

		cudaMemcpy(res,d_res,out_size,cudaMemcpyDeviceToHost);

		cudaFree(d_in);
		cudaFree(d_res);

	}

	void launch_conv2d(float *input, float *filter, float * out,const int batch_size, const int stride, const int input_H, const int input_W, 
						const int filter_H, const int filter_W, 
						const int out_H,const int out_W,
						const int in_channel, const int out_channel,cudaStream_t stream)
	{
		int input_size=sizeof(float)*input_H*input_W*in_channel*batch_size;
		int filter_size=sizeof(float)*filter_H*filter_W*out_channel*in_channel;
		int out_size=sizeof(float)*out_H*out_W*out_channel*batch_size;
		int shift_size=sizeof(float)*batch_size*in_channel*input_H*input_W;

		float *d_in, *d_filter, *d_out, *temp, *shift;
		cudaMalloc((void**)&d_in,input_size);
		cudaMalloc((void**)&d_filter,filter_size);
		cudaMalloc((void**)&d_out,out_size);
		cudaMalloc((void**)&temp,out_size);

		cudaMalloc((void**)&shift,shift_size);


		cudaMemcpy(d_in,input,input_size,cudaMemcpyHostToDevice);
		cudaMemcpy(d_filter,filter,filter_size,cudaMemcpyHostToDevice);
		cudaMemcpy(d_out,out,out_size,cudaMemcpyHostToDevice);
		cudaMemcpy(temp,out,out_size,cudaMemcpyHostToDevice);

		dim3 grid_dim(batch_size);
		dim3 block_dim(32,32,1);

		int nthreadx=(out_H*out_W+TILE_SIZE-1)/TILE_SIZE;
		int nthready=(out_channel+TILE_SIZE-1)/TILE_SIZE;
		dim3 block_dim2(nthreadx,nthready);


		int i,j;
		for(i=0;i<filter_H;i++)
		{
			for(j=0;j<filter_W;j++)
			{
				//读取input的一部分
				shift_kernel<<<grid_dim,block_dim,0,stream>>>(d_in,shift,i,j,stride,input_H,input_W,out_H,out_W,in_channel);
				//和filter的[i,j]个元素做bmm, 并加给d_out
				conv_bmm<<<grid_dim,block_dim2,0,stream>>>(shift,d_filter+i*filter_W*in_channel*out_channel+j*in_channel*out_channel,d_out,batch_size,out_H*out_W,in_channel,out_channel);
			}

		}
		
		cudaMemcpy(out,d_out,out_size,cudaMemcpyDeviceToHost);

		cudaFree(d_in);
		cudaFree(d_filter);
		cudaFree(d_out);
		cudaFree(temp);
		cudaFree(shift);


	}
}