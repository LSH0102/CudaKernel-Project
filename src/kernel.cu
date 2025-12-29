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

	//每个thread处理一个 tile_size x tile_size 的小矩阵块 i.e左上角为 tile_size *i,tile_size *j 的矩阵块
	
	int iters=(m+31)/32;
	int iters2=(p+31)/32;
	int tile_size=iters;
	if (iters2>iters)
	{
		tile_size=iters2;
	}

	int i=threadIdx.x;
	int j=threadIdx.y;
	int k=0;
	int row=0;
	int column=0;
	float temp=0.0;
	//计算c中第row行, column列的结果
	for(row=i*tile_size;row<m&& row<(i+1)*tile_size;row++)
	{
		for(column=j*tile_size;column<p&&column<(j+1)*tile_size;column++)
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

	//每个thread处理一个小矩阵块 i.e左上角为 tile_size *i,tile_size *j 的矩阵块

	int iters=(m+31)/32;
	int iters2=(p+31)/32;
	int tile_size=iters;
	if (iters2>iters)
	{
		tile_size=iters2;
	}
	
	int i=threadIdx.x;
	int j=threadIdx.y;
	int k=0;
	int row=0;
	int column=0;
	float temp=0.0;
	//计算c中第row行, column列的结果
	for(row=i*tile_size;row<m&& row<(i+1)*tile_size;row++)
	{
		for(column=j*tile_size;column<p&&column<(j+1)*tile_size;column++)
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
__global__ void flash_attention_forward(float *dQ, float *dK, float *dV,float *dm, float *dO,  int batch_size, int q_len, int kv_len, int d, float scale, int Tq, int Tkv)
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
			Sij[threadIdx.x][threadIdx.y]=temp/scale+dm[(Bq*i+threadIdx.x)*kv_len+Bkv*j+threadIdx.y];
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
///flash attention backward的rowsum(dO*O)
__global__ void rowsum(float *dO, float *O, float *D,int batch_size, int q_len, int d,int Tq)
{
	//block的总数是batch_size*Tq, batch_id*Tq+i处理第batch_id个batch的第i个tile
	
	int batch_id=blockIdx.x/Tq;
	int i=blockIdx.x-batch_id*Tq;

	//这个block需要处理一个瓦片 Qij=(Bq,d)
	int j=0;
	
	//先读取Qi并初始化li和mi
	if(threadIdx.x<Bq && threadIdx.y==0 && Bq*i+threadIdx.x<q_len)
	{
		D[batch_id*q_len+(Bq*i+threadIdx.x)]=0.0;
		for(j=0;j<d;j++)
		{
			D[batch_id*q_len+(Bq*i+threadIdx.x)]+=dO[batch_id*q_len*d+(Bq*i+threadIdx.x)*d+j]*O[batch_id*q_len*d+(Bq*i+threadIdx.x)*d+j];
			
		}
	}
}

///flash attention bachward 的主内核函数
__global__ void flash_attention_backward(float *Q, float* O,float *dO, float* K, float *V, float *dQ, float *dK, 
										float *dV,float *L,float *D, float *m, int batch_size, int q_len,int kv_len, int d, float scale,int Tq, int Tkv)
{
	//block的总数是batch_size*Tkv, batch_id*Tkv+j处理第batch_id个batch的第j个tile
	
	int batch_id=blockIdx.x/Tkv;
	int j=blockIdx.x-batch_id*Tkv;
	int i,k,t;
	float temp;
	//这个block需要处理一个瓦片 Sij=(Bq,Bkv) for 1\leq i \leq Tq

	//声明block内部的共享数据
	__shared__ float Qi[Bq][128];
	__shared__ float dOi[Bq][128];

	__shared__ float li[Bq];
	__shared__ float Di[Bq];
	__shared__ float Kj[Bkv][128];
	__shared__ float Vj[Bkv][128];
	__shared__ float dKj[Bkv][128];
	__shared__ float dVj[Bkv][128];
	__shared__ float Sij[Bq][Bkv];
	__shared__ float dPij[Bq][Bkv];

	//读取Kj和Vj并初始化dKj和dVj
	if(threadIdx.x<Bkv && threadIdx.y==0 && Bkv*j+threadIdx.x<kv_len)
	{
		for(i=0;i<d;i++)
		{
			Kj[threadIdx.x][i]=K[batch_id*kv_len*d+(Bkv*j+threadIdx.x)*d+i];
			Vj[threadIdx.x][i]=V[batch_id*kv_len*d+(Bkv*j+threadIdx.x)*d+i];
			dKj[threadIdx.x][i]=0.0;
			dVj[threadIdx.x][i]=0.0;
		}
	}
	__syncthreads();

	for(i=0;i<Tq;i++)
	{
		//加载Qi,Oi,dQi,dOi,Li,Di;
		if(threadIdx.x==0&&threadIdx.y<Bq && Bq*i+threadIdx.y<q_len)
		{
			for(k=0;k<d;k++)
			{
				Qi[threadIdx.y][k]=Q[batch_id*q_len*d+(Bq*i+threadIdx.y)*d+k];
				dOi[threadIdx.y][k]=dO[batch_id*q_len*d+(Bq*i+threadIdx.y)*d+k];
			}
			li[threadIdx.y]=L[batch_id*q_len+(Bq*i+threadIdx.y)];
			Di[threadIdx.y]=D[batch_id*q_len+(Bq*i+threadIdx.y)];
		}
		__syncthreads();
		//计算矩阵乘法然后直接转化为注意力概率
		temp=0.0;
		if(threadIdx.x<Bq && threadIdx.y<Bkv && Bkv*j+threadIdx.y<kv_len&& Bq*i+threadIdx.x<q_len)
		{
			for(k=0;k<d;k++)
			{
				temp+=Qi[threadIdx.x][k]*Kj[threadIdx.y][k];
			
			}
			Sij[threadIdx.x][threadIdx.y]=__expf(  temp/scale+m[(Bq*i+threadIdx.x)*kv_len+Bkv*j+threadIdx.y]-li[threadIdx.x]);
		}
		__syncthreads();

		//更新dV, 这是矩阵乘法
		if(threadIdx.x<Bkv&& Bkv*j+threadIdx.x<kv_len && threadIdx.y==0)
		{
			for(k=0;k<d;k++)
			{
				for(t=0;t<Bq && Bq*i+t<q_len;t++)
				{
					dVj[threadIdx.x][k]+=Sij[t][threadIdx.x]*dOi[t][k];
				}
			}
		}
		__syncthreads();

		//计算dPij
		temp=0.0;
		if(threadIdx.x<Bq && threadIdx.y<Bkv && Bkv*j+threadIdx.y<kv_len&& Bq*i+threadIdx.x<q_len)
		{
			for(k=0;k<d;k++)
			{
				temp+=dOi[threadIdx.x][k]*Vj[threadIdx.y][k];
			
			}
			dPij[threadIdx.x][threadIdx.y]=temp;
		}
		__syncthreads();
		//计算dSij
		if(threadIdx.x<Bq && threadIdx.y<Bkv && Bkv*j+threadIdx.y<kv_len&& Bq*i+threadIdx.x<q_len)
		{
			
			Sij[threadIdx.x][threadIdx.y]=Sij[threadIdx.x][threadIdx.y]*(dPij[threadIdx.x][threadIdx.y]-Di[threadIdx.x])/scale;
		}
		__syncthreads();

		//原子更新dQ 
		if(threadIdx.x<Bq&& Bq*i+threadIdx.x<q_len && threadIdx.y==0)
		{
			for(k=0;k<d;k++)
			{
				temp=0.0;
				for(t=0;t<Bkv && Bkv*j+t<kv_len;t++)
				{
					temp+=Sij[threadIdx.x][t]*Kj[t][k];
				}
				atomicAdd(&dQ[batch_id*q_len*d+(Bq*i+threadIdx.x)*d+k],temp);
			}
		}
		__syncthreads();

		//更新dKj
		if(threadIdx.x<Bkv&& Bkv*j+threadIdx.x<kv_len && threadIdx.y==0)
		{
			for(k=0;k<d;k++)
			{
				for(t=0;t<Bq && Bq*i+t<q_len;t++)
				{
					dKj[threadIdx.x][k]+=Sij[t][threadIdx.x]*Qi[t][k];
				}
			}
		}
		__syncthreads();
		//开始下一轮循环

	}
	//将dK和dQ写回存储
	if(threadIdx.x<Bkv && Bkv*j+threadIdx.x<kv_len)
	{
		if(threadIdx.y==0)
		{
			for(k=0;k<d;k++)
			{
			
				dK[batch_id*kv_len*d+(Bkv*j+threadIdx.x)*d+k]=dKj[threadIdx.x][k];
				dV[batch_id*kv_len*d+(Bkv*j+threadIdx.x)*d+k]=dVj[threadIdx.x][k];
			
			}
		}
	}
}

///conv2d的backward的翻转函数
__global__ void filp_and_transpose(float *filter, float* out, int h, int w, int in_channel, int out_channel)
{
	int channel_id=blockIdx.x;
	int i,j,k;
	int row_id=threadIdx.x;
	int col_id=threadIdx.y;

	int num1=(h+31)/32;
	int num2=(w+31)/32;
	if(channel_id>=in_channel)
	{
		return;
	}
	for(k=0;k<out_channel;k++)
	{
		//一个tread处理4个
		for(i=0;i<num1 && row_id*num1+i<h;i++)
		{
			for(j=0;j<num2 && col_id*num2+j<w;j++)
			{
				out[(h-1-row_id*num1-i)*w*in_channel*out_channel+(w-1-j-col_id*num2)*in_channel*out_channel+k*in_channel+channel_id]=filter[(row_id*num1+i)*w*in_channel*out_channel+(col_id*num2+j)*in_channel*out_channel+channel_id*out_channel+k];
			}
		}
	}
}

///conv2d用到的填充函数
__global__ void fill_dO (float *d_o, float *res, int filter_H, int filter_W, int out_H, int out_W,int input_H, int input_W, int stride, int out_channel, int batch_size)
{
	int batch_id=blockIdx.x;
	int channel_id=blockIdx.y;

	int num1=(out_H+31)/32;
	int num2=(out_W+31)/32;

	int i,j;
	int row_id=threadIdx.x;
	int col_id=threadIdx.y;
	int tar_x,tar_y;
	for(i=0;i<num1 && row_id*num1+i<out_H;i++)
	{
		for(j=0;j<num2 && col_id*num2+j<out_W;j++)
		{
			tar_x=filter_H-1+(row_id*num1+i)*stride;
			tar_y=filter_W-1+(col_id*num2+j)*stride;
			res[batch_id*(filter_H+input_H-1)*(filter_W+input_W-1)*out_channel+tar_x*(filter_W+input_W-1)*out_channel+tar_y*out_channel+channel_id]
					=d_o[batch_id*out_H*out_W*out_channel+(row_id*num1+i)*out_W*out_channel+(col_id*num2+j)*out_channel+channel_id];
			
		}
	}
}

///conv2d计算dw的函数
__global__ void compute_dw(float *x,float *dO, float *dw,int batch_size, int stride,
							int input_H, int input_W,int filter_H,int filter_W,int out_H,int out_W,int in_channel,int out_channel, int p,int q)
{
	int in_channel_id=blockIdx.x*32+threadIdx.x;
	int out_channel_id=blockIdx.y*32+threadIdx.y;
	if(in_channel_id>=in_channel || out_channel_id>=out_channel)
	{
		return;
	}
	int i,j,k;
	for(k=0;k<batch_size;k++)
	{
		for(i=0;i<out_H;i++)
		{
			for(j=0;j<out_W;j++)
			{
				dw[p*filter_W*in_channel*out_channel+q*in_channel*out_channel+in_channel_id*out_channel+out_channel_id]+=
						dO[k*out_H*out_W*out_channel+i*out_W*out_channel+j*out_channel+out_channel_id]*x[k*input_H*input_W*in_channel+(p+i*stride)*input_W*in_channel+(q+j*stride)*in_channel+in_channel_id];
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

		
		
		dim3 grid_dim(batch_size);
		dim3 block_dim(32,32,1);

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

		
		dim3 block_dim2(32,32,1);


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

	void device_conv2d(float *input, float *filter, float * out,const int batch_size, const int stride, const int input_H, const int input_W, 
						const int filter_H, const int filter_W, 
						const int out_H,const int out_W,
						const int in_channel, const int out_channel,cudaStream_t stream)
	{
		
		int shift_size=sizeof(float)*batch_size*in_channel*input_H*input_W;
		float *shift;
		cudaMalloc((void**)&shift,shift_size);

		dim3 grid_dim(batch_size);
		dim3 block_dim(32,32,1);


		
		dim3 block_dim2(32,32,1);


		int i,j;
		for(i=0;i<filter_H;i++)
		{
			for(j=0;j<filter_W;j++)
			{
				//读取input的一部分
				shift_kernel<<<grid_dim,block_dim,0,stream>>>(input,shift,i,j,stride,input_H,input_W,out_H,out_W,in_channel);
				//和filter的[i,j]个元素做bmm, 并加给d_out
				conv_bmm<<<grid_dim,block_dim2,0,stream>>>(shift,filter+i*filter_W*in_channel*out_channel+j*in_channel*out_channel,out,batch_size,out_H*out_W,in_channel,out_channel);
			}

		}
		cudaFree(shift);


	}
	
	void launch_flash_attention_forward(float *Q, float *K, float *V,float *m, float *O,  int batch_size, int q_len, int kv_len, int d, float scale, cudaStream_t stream)
	{
		int Q_size=sizeof(float)*batch_size*q_len*d;
		int KV_size=sizeof(float)*batch_size*kv_len*d;
		int O_size=sizeof(float)*batch_size*q_len*d;
		int m_size=sizeof(float)*q_len*kv_len;

		float *d_Q, *d_K, *d_V, *d_O,*d_m;
		cudaMalloc((void**)&d_Q,Q_size);
		cudaMalloc((void**)&d_K,KV_size);
		cudaMalloc((void**)&d_V,KV_size);
		cudaMalloc((void**)&d_O,O_size);
		cudaMalloc((void**)&d_m,m_size);

		cudaMemcpy(d_Q,Q,Q_size,cudaMemcpyHostToDevice);
		cudaMemcpy(d_K,K,KV_size,cudaMemcpyHostToDevice);
		cudaMemcpy(d_V,V,KV_size,cudaMemcpyHostToDevice);
		cudaMemcpy(d_O,O,O_size,cudaMemcpyHostToDevice);
		cudaMemcpy(d_m,m,m_size,cudaMemcpyHostToDevice);

		//把Q切成(batch,Bq,d), KV切成(batch,Bkv,d)

		int Tq=(q_len+Bq-1)/Bq;
		int Tkv=(kv_len+Bkv-1)/Bkv;

		
		dim3 grid_dim(batch_size*Tq,1,1);
		dim3 block_dim(Bq,Bkv,1);
		flash_attention_forward<<<grid_dim,block_dim,0,stream>>>(d_Q,d_K,d_V,d_m,d_O, batch_size,q_len,kv_len,d,scale,Tq,Tkv);


		cudaMemcpy(O,d_O,O_size,cudaMemcpyDeviceToHost);

		cudaFree(d_Q);
		cudaFree(d_K);
		cudaFree(d_V);
		cudaFree(d_O);
		cudaFree(d_m);
	}

	void launch_flash_attention_backward(float *Q, float *K, float *V, float *O,float *dO, float *L,
										float *pQ, float *pK, float *pV,float *m,
	
										int batch_size, int q_len, int kv_len, int d, float scale, cudaStream_t stream)
	{
		int Q_size=sizeof(float)*batch_size*q_len*d;
		int KV_size=sizeof(float)*batch_size*kv_len*d;
		int O_size=sizeof(float)*batch_size*q_len*d;
		int L_size=sizeof(float)*batch_size*q_len;
		int m_size=sizeof(float)*q_len*kv_len;

		float *d_Q, *d_K, *d_V, *d_O, *ddO, *d_L, *d_D, *d_pQ, *d_pK, *d_pV, *d_m;
		cudaMalloc((void**)&d_Q,Q_size);
		cudaMalloc((void**)&d_K,KV_size);
		cudaMalloc((void**)&d_V,KV_size);
		cudaMalloc((void**)&d_O,O_size);
		cudaMalloc((void**)&ddO,O_size);
		cudaMalloc((void**)&d_L,L_size);
		cudaMalloc((void**)&d_D,L_size);
		cudaMalloc((void**)&d_pQ,Q_size);
		cudaMalloc((void**)&d_pK,KV_size);
		cudaMalloc((void**)&d_pV,KV_size);
		cudaMalloc((void**)&d_m,m_size);

		cudaMemcpy(d_Q,Q,Q_size,cudaMemcpyHostToDevice);
		cudaMemcpy(d_K,K,KV_size,cudaMemcpyHostToDevice);
		cudaMemcpy(d_V,V,KV_size,cudaMemcpyHostToDevice);
		cudaMemcpy(d_O,O,O_size,cudaMemcpyHostToDevice);
		cudaMemcpy(ddO,dO,O_size,cudaMemcpyHostToDevice);
		cudaMemcpy(d_L,L,L_size,cudaMemcpyHostToDevice);
		cudaMemcpy(d_pQ,pQ,Q_size,cudaMemcpyHostToDevice);
		cudaMemcpy(d_pK,pK,KV_size,cudaMemcpyHostToDevice);
		cudaMemcpy(d_pV,pV,KV_size,cudaMemcpyHostToDevice);
		cudaMemcpy(d_m,m,m_size,cudaMemcpyHostToDevice);

		int Tq=(q_len+Bq-1)/Bq;
		int Tkv=(kv_len+Bkv-1)/Bkv;

		
		dim3 grid_dim(batch_size*Tq,1,1);
		dim3 grid_dim2(batch_size*Tkv,1,1);
		dim3 block_dim(Bq,Bkv,1);

		rowsum<<<grid_dim,block_dim,0,stream>>>(ddO, d_O, d_D,batch_size,q_len,d,Tq);
		
		flash_attention_backward<<<grid_dim2,block_dim,0,stream>>>(d_Q,d_O,ddO,d_K,d_V,d_pQ,d_pK,d_pV,d_L,d_D,d_m,batch_size,q_len,kv_len,d,scale,Tq,Tkv);

		cudaMemcpy(pK,d_pK,KV_size,cudaMemcpyDeviceToHost);
		cudaMemcpy(pV,d_pV,KV_size,cudaMemcpyDeviceToHost);
		cudaMemcpy(pQ,d_pQ,Q_size,cudaMemcpyDeviceToHost);

		cudaFree(d_Q);
		cudaFree(d_K);
		cudaFree(d_V);
		cudaFree(d_O);
		cudaFree(ddO);
		cudaFree(d_L);
		cudaFree(d_D);
		cudaFree(d_pQ);
		cudaFree(d_pK);
		cudaFree(d_pV);
		cudaFree(d_m);
	}

	void launch_conv2d_backward(float *input, float *filter, float *out_grad, float *dx, float *dw,const int batch_size, const int stride, 
						const int input_H, const int input_W, const int filter_H,const int filter_W, 
						const int out_H, const int out_W,const int in_channel, const int out_channel,cudaStream_t stream)
	{
		int input_size=sizeof(float)*input_H*input_W*in_channel*batch_size;
		int filter_size=sizeof(float)*filter_H*filter_W*out_channel*in_channel;
		int out_size=sizeof(float)*out_H*out_W*out_channel*batch_size;

		int dO_size=sizeof(float)*out_channel*(input_H+filter_H-1)*(input_W+filter_W-1)*batch_size;
		
		float *tilde_w,*d_filter, *dO, *d_x,*d_o, *d_input;
		cudaMalloc((void**)&tilde_w,filter_size);
		cudaMalloc((void**)&d_filter,filter_size);
		cudaMalloc((void**)&dO,dO_size);
		cudaMalloc((void**)&d_x,input_size);
		cudaMalloc((void**)&d_o,out_size);
		cudaMalloc((void**)&d_input,input_size);



		cudaMemcpy(d_filter,filter,filter_size,cudaMemcpyHostToDevice);
		cudaMemcpy(d_o,out_grad,out_size,cudaMemcpyHostToDevice);
		cudaMemcpy(tilde_w,filter,filter_size,cudaMemcpyHostToDevice);
		cudaMemcpy(d_input,input,input_size,cudaMemcpyHostToDevice);

		//对w进行前两个维数的翻转和后两个维数的转置
		dim3 grid_dim1(in_channel);
		dim3 block_dim1(32,32,1);
		filp_and_transpose<<<grid_dim1,block_dim1,0,stream>>>(d_filter,  tilde_w,  filter_H,  filter_W, in_channel,out_channel);
		cudaDeviceSynchronize();

		//填充dO
		dim3 grid_dim2(batch_size,out_channel,1);
		dim3 block_dim2(32,32,1);
		fill_dO<<<grid_dim2,block_dim2,0,stream>>>(d_o, dO, filter_H, filter_W, out_H, out_W,input_H, input_W,stride, out_channel, batch_size );
		cudaDeviceSynchronize();


		device_conv2d(dO, tilde_w, d_x, batch_size, 1, input_H+filter_H-1, input_W+filter_W-1, 
						filter_H, filter_W, 
						input_H,input_W,
						out_channel, in_channel,stream);
		

		cudaDeviceSynchronize();
		cudaMemcpy(dx,d_x,input_size,cudaMemcpyDeviceToHost);
		
		//然后计算dw
		cudaMemcpy(d_filter,dw,filter_size,cudaMemcpyHostToDevice);

		dim3 grid_dim3((in_channel+31)/32,(out_channel+31)/32,1);
		dim3 block_dim3(32,32,1);
		int p,q;
		for(p=0;p<filter_H;p++)
		{
			for(q=0;q<filter_W;q++)
			{
				compute_dw<<<grid_dim3,block_dim3,0,stream>>>(d_input,d_o, d_filter,batch_size,stride,input_H, input_W, filter_H, filter_W, out_H, out_W, in_channel, out_channel,p,q);
			}
		}

		cudaMemcpy(dw,d_filter,filter_size,cudaMemcpyDeviceToHost);
		
		cudaFree(tilde_w);
		cudaFree(d_filter);
		cudaFree(dO);
		cudaFree(d_x);
		cudaFree(d_o);
		cudaFree(d_input);
	}

}