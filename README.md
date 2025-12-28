# My project of writing cuda kernels
Used only for study and communication purpose

To compile kernels, use the following command

`nvcc -Xcompiler -fPIC -shared src/kernel.cu -o cuda_kernels/kernel.so`

then 

`python test.py`

## Functions
1. Vector addition (added on 2025/12/1)
2. batched matrix multiplication (added on 2025/12/2)
3. conv2d (added on 2025/12/13)
4. flash attention forward (added on 2025/12/15)
5. flash attention backward (added on 2025/12/16)

### Conv2d:implementation
A conv2d operation with high performance can be regarded as a series of matrix multiplication. 

$$Out[b,i,j,t]=\sum_{m=0}^{filter\textunderscore H-1}\sum_{n=0}^{filter\textunderscore W-1}\sum_{c=0}^{in\textunderscore channel-1}x[b,i+m,j+n,c]\cdot w[m,n,c,t]$$

So, it sufficies to get x[:,i:i+height,j:j+width,:] and reshape to (batch_size,out_height*out_width, in_channel) and invoke batched matrix multiplication with $w[m,n]\in R^{in\textunderscore channel \times out\textunderscore channel }$

Here is a numpy implementation of conv2d:
```python
def conv_np(x,w,stride,padding):

    batch_size=x.shape[0]
    in_channel=x.shape[-1]
    width=x.shape[2]
    height=x.shape[1]
    out_channel=w.shape[-1]
    o_height=(2*padding+height-w.shape[0])//stride+1
    o_width=(2*padding+width-w.shape[1])//stride+1
    out=np.zeros((batch_size,o_height,o_width,out_channel))
    x=np.pad(x, pad_width=((0,0),(padding,padding),(padding,padding),(0,0)))
    
    for i in range(0,w.shape[0]):
        for j in range(0,w.shape[1]):
            shift=x[:,i:i+o_height*stride:stride,j:j+o_width*stride:stride,:]
            shift=shift.reshape((batch_size,o_width*o_height,in_channel))
            filt=w[i,j].reshape((1,in_channel,out_channel))
            out+=np.matmul(shift,filt).reshape((batch_size,o_height,o_width,out_channel))
    return out 
```

### Conv2d backward:

$$Out[b,i,j,t]=\sum_{m=0}^{filter\textunderscore H-1}\sum_{n=0}^{filter\textunderscore W-1}\sum_{c=0}^{in\textunderscore channel-1}x[b,i* stride+m,j* stride+n,c]\cdot w[m,n,c,t]$$

Thus we have 
$$\frac{\partial L}{\partial x[b,i,j,t]}=\sum_{u,v,w,y}dO[u,v,w,y]\frac{\partial O[u,v,w,y]}{\partial x[b,i,j,t]}$$

So the above term $\neq 0$ if and only if $u=b,c=t,i=v * stride+m, j=w* stride+n$, i.e 

$$\frac{\partial L}{\partial x[b,i,j,t]}=\sum_{m,n}\sum_{y}dO[b,\frac{i-m}{stride},\frac{j-n}{stride},y] * w[m,n,c,y]$$
