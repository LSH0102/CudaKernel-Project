#include <cuda.h>
#include <cstddef>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

const int TILE_SIZE=64;
const int Bq=4;
const int Bkv=4;
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

///flash attention forward的内核
__global__ void flash_attention_forward(float *dQ, float *dK, float *dV, float *dO, int batch_size, int q_len, int kv_len, int d, float scale, int Tq, int Tkv)
{
	//block的总数是batch_size*Tq, batch_id*Tq+i处理第batch_id个batch的第i个tile
	
	int batch_id=blockIdx.x/Tq;
	int i=blockIdx.x-batch_id*Tq;

	//这个block需要处理一个瓦片 Sij=(Bq,Bkv) for 1\leq j\leq Tkv
	int j=0;
	int k=0;

	//声明block内部的共享数据
	__shared__ float Qi[Bq][128];
	__shared__ float Oij[Bq][128];
	__shared__ float li[Bq];
	__shared__ float mi[Bq];
	__shared__ float Kj[Bkv][128];
	__shared__ float Vj[Bkv][128];
	__shared__ float Sij[Bq][Bkv];
	__shared__ float new_mi[Bq];

	__shared__ float row_reduction[Bq][128];

	
	float temp=0.0;
	int step=1;
	int t=0;
	//先读取Qi并初始化li和mi
	if(threadIdx.x<Bq && threadIdx.y==0 && Bq*i+threadIdx.x<q_len)
	{
		for(j=0;j<d;j++)
		{
			Qi[threadIdx.x][j]=dQ[batch_id*q_len*d+(Bq*i+threadIdx.x)*d+j];
			
		}
		li[threadIdx.x]=0.0;
		mi[threadIdx.x]=-INFINITY;
	}
	__syncthreads();
	for(j=0;j<Tkv;j++)
	{
		//读取Kj和Vj
		if(threadIdx.x==0&&threadIdx.y<Bkv && Bkv*j+threadIdx.y<kv_len)
		{
			for(k=0;k<d;k++)
			{
				Kj[threadIdx.y][k]=dK[batch_id*kv_len*d+(Bkv*j+threadIdx.y)*d+k];
				Vj[threadIdx.y][k]=dV[batch_id*kv_len*d+(Bkv*j+threadIdx.y)*d+k];
				
			}
		}
		__syncthreads();
		//准备矩阵乘法并除以缩放因子scale
		temp=0.0;
		if(threadIdx.x<Bq && threadIdx.y<Bkv && Bkv*j+threadIdx.y<kv_len&& Bq*i+threadIdx.x<q_len)
		{
			for(k=0;k<d;k++)
			{
				temp+=Qi[threadIdx.x][k]*Kj[threadIdx.y][k];
			
			}
			Sij[threadIdx.x][threadIdx.y]=temp/scale;
			row_reduction[threadIdx.x][threadIdx.y]=Sij[threadIdx.x][threadIdx.y];
		}
		
		//求Sij的逐行最大值并放入row_reduction
		if(threadIdx.x<Bq && Bq*i+threadIdx.x<q_len)
		{
			for(step=2;step<4*d;step=2*step)
			{
				if(threadIdx.y%step==0&&threadIdx.y+step/2<Bkv && threadIdx.y+step/2<kv_len)
				{
					if(row_reduction[threadIdx.x][threadIdx.y]>=row_reduction[threadIdx.x][threadIdx.y+step/2])
					{
						row_reduction[threadIdx.x][threadIdx.y]=row_reduction[threadIdx.x][threadIdx.y];
					}
					else
					{
						row_reduction[threadIdx.x][threadIdx.y]=row_reduction[threadIdx.x][threadIdx.y+step/2];
					}
				}
			}	
		}
		__syncthreads();

		
		//求mi^{j}=max ( mi^{j-1},rowmaxSij)
		if(threadIdx.y==0 && threadIdx.x<Bq && Bq*i+threadIdx.x<q_len)
		{
			if(mi[threadIdx.x]<row_reduction[threadIdx.x][0])
			{
				new_mi[threadIdx.x]=row_reduction[threadIdx.x][0];
			}
			else
			{
				new_mi[threadIdx.x]=mi[threadIdx.x];
			}
		}
		__syncthreads();

		//此时的Sij是Pij. 将row_reduction初始化, 准备进行逐行求和
		if(threadIdx.x<Bq && threadIdx.y<Bkv && Bkv*j+threadIdx.y<kv_len&& Bq*i+threadIdx.x<q_len)
		{
			Sij[threadIdx.x][threadIdx.y]=__expf( Sij[threadIdx.x][threadIdx.y]-new_mi[threadIdx.x] );
			row_reduction[threadIdx.x][threadIdx.y]=Sij[threadIdx.x][threadIdx.y];
		}
		__syncthreads();

		//计算Pij的逐行和
		if(threadIdx.x<Bq && Bq*i+threadIdx.x<q_len)
		{
			for(step=2;step<4*d;step=2*step)
			{
				if(threadIdx.y%step==0&&threadIdx.y+step/2<Bkv && threadIdx.y+step/2<kv_len)
				{
					row_reduction[threadIdx.x][threadIdx.y]+=row_reduction[threadIdx.x][threadIdx.y+step/2];
				}
			}	
		}
		__syncthreads();

		if(threadIdx.x<Bq && Bq*i+threadIdx.x<q_len)
		{
			if(threadIdx.y==0)
			{
				li[threadIdx.x]=__expf(mi[threadIdx.x]-new_mi[threadIdx.x] )*li[threadIdx.x]+row_reduction[threadIdx.x][0];
				for(t=0;t<d;t++)
				{
					Oij[threadIdx.x][t]=Oij[threadIdx.x][t]*__expf(mi[threadIdx.x]-new_mi[threadIdx.x]);
				}
			}
		}
		__syncthreads();

		//加上Pij乘以 Vj
		if(threadIdx.x<Bq && Bq*i+threadIdx.x<q_len)
		{
			if(threadIdx.y==0)
			{
				for(t=0;t<d;t++)
				{
					for(k=0;k<Bkv;k++)
					{
						Oij[threadIdx.x][t]+=Sij[threadIdx.x][k]*Vj[k][t];	
					}
				}
				//更新mi, 因为mi没法做到原地更新
				mi[threadIdx.x]=new_mi[threadIdx.x];
			
			}
		}
		__syncthreads();
		//开启下一轮循环
	}

	//准备将对应的Oij写入输出
	if(threadIdx.x<Bq && Bq*i+threadIdx.x<q_len)
	{
		if(threadIdx.y==0)
		{
			for(k=0;k<d;k++)
			{
			
				dO[batch_id*q_len*d+(Bq*i+threadIdx.x)*d+k]=Oij[threadIdx.x][k]/li[threadIdx.x];
			
			}
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
	
	void launch_flash_attention_forward(float *Q, float *K, float *V, float *O, int batch_size, int q_len, int kv_len, int d, float scale, cudaStream_t stream)
	{
		int Q_size=sizeof(float)*batch_size*q_len*d;
		int KV_size=sizeof(float)*batch_size*kv_len*d;
		int O_size=sizeof(float)*batch_size*q_len*d;

		float *d_Q, *d_K, *d_V, *d_O;
		cudaMalloc((void**)&d_Q,Q_size);
		cudaMalloc((void**)&d_K,KV_size);
		cudaMalloc((void**)&d_V,KV_size);
		cudaMalloc((void**)&d_O,O_size);

		cudaMemcpy(d_Q,Q,Q_size,cudaMemcpyHostToDevice);
		cudaMemcpy(d_K,K,KV_size,cudaMemcpyHostToDevice);
		cudaMemcpy(d_V,V,KV_size,cudaMemcpyHostToDevice);
		cudaMemcpy(d_O,O,O_size,cudaMemcpyHostToDevice);


		//把Q切成(batch,Bq,d), KV切成(batch,Bkv,d)

		int Tq=(q_len+Bq-1)/Bq;
		int Tkv=(kv_len+Bkv-1)/Bkv;

		
		dim3 grid_dim(batch_size*Tq,1,1);
		dim3 block_dim(Bq,Bkv,1);
		flash_attention_forward<<<grid_dim,block_dim,0,stream>>>(d_Q,d_K,d_V,d_O,batch_size,q_len,kv_len,d,scale,Tq,Tkv);


		cudaMemcpy(O,d_O,O_size,cudaMemcpyDeviceToHost);

		cudaFree(d_Q);
		cudaFree(d_K);
		cudaFree(d_V);
		cudaFree(d_O);


	}
}