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

### Conv2d:implementation
A conv2d operation with high performance can be regarded as a series of matrix multiplication. Here is a numpy implementation of conv2d:

`def conv_np(x,w,stride,padding):
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
    return out `
