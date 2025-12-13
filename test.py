# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 18:49:00 2025

@author: 25488
"""
import ctypes
import torch
import numpy as np
import time 
lib=ctypes.CDLL("cuda_kernels/kernel.so")
datatype=np.float32

def vector_add(a:torch.Tensor,b:torch.Tensor):
    size=a.numel()
    shape=a.shape
    c=torch.zeros_like(a).view(-1)
    stream=torch.cuda.current_stream().cuda_stream
    
    
    lib.launch_add.argtypes=[
        
        np.ctypeslib.ndpointer(dtype=datatype,ndim=1,flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype,ndim=1,flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype,ndim=1,flags='C_CONTIGUOUS'),
        ctypes.c_int,
        ctypes.c_void_p]
    
    lib.launch_add.restype=None
    
    res=c.view(-1).cpu().numpy()
    lib.launch_add(a.view(-1).cpu().numpy(),b.view(-1).cpu().numpy(),res,size,stream)
    torch.cuda.synchronize()
    
    c=torch.from_numpy(res).to(a.device)
    c=c.reshape(shape)
    return c

def bmm(a:torch.Tensor,b:torch.Tensor):
    batch_size,m,n=a.shape
    p=b.shape[-1]
    
    res=torch.zeros(size=(batch_size,m,p)).view(-1).numpy()
    stream=torch.cuda.current_stream().cuda_stream
    
    lib.launch_bmm.argtypes=[
        
        np.ctypeslib.ndpointer(dtype=datatype,ndim=1,flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype,ndim=1,flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype,ndim=1,flags='C_CONTIGUOUS'),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p]
    lib.launch_bmm.restype=None
    
    lib.launch_bmm(a.view(-1).cpu().numpy(),b.view(-1).cpu().numpy(),res, batch_size,m,n,p,stream)
    c=torch.from_numpy(res).to(a.device)
    c=c.reshape((batch_size,m,p))
    
    return c

def conv2d(a:np.ndarray,w:np.ndarray,stride,padding):
    
    batch_size=a.shape[0]
    in_channel=a.shape[-1]
    width=a.shape[2]
    height=a.shape[1]
    out_channel=w.shape[-1]
    
    o_height=(2*padding+height-w.shape[0])//stride+1
    o_width=(2*padding+width-w.shape[1])//stride+1
    
    a=np.pad(a, pad_width=((0,0),(padding,padding),(padding,padding),(0,0)))
    
    res=torch.zeros(size=(batch_size,o_height,o_width,out_channel)).view(-1).numpy()
    stream=torch.cuda.current_stream().cuda_stream
    
    lib.launch_conv2d.argtypes=[
        np.ctypeslib.ndpointer(dtype=datatype,ndim=1,flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype,ndim=1,flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype,ndim=1,flags='C_CONTIGUOUS'),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p]
    
    lib.launch_conv2d.restype=None
    
    lib.launch_conv2d(a.flatten(),w.flatten(),res,batch_size,stride,height+2*padding,width+2*padding,w.shape[0],w.shape[1],o_height,o_width,in_channel,out_channel,stream)
    
    res=res.reshape((batch_size,o_height,o_width,out_channel))
    return res
    
    

def shift(a:np.ndarray,i,j,o_height,o_width,stride):
    b,h,w,c=a.shape
    res=torch.zeros(size=(b,o_height,o_width,c)).view(-1).numpy()
    
    stream=torch.cuda.current_stream().cuda_stream
    
    lib.launch_shift.argtypes=[
        
        np.ctypeslib.ndpointer(dtype=datatype,ndim=1,flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype,ndim=1,flags='C_CONTIGUOUS'),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p]
    
    lib.launch_shift.restype=None
    
    
    lib.launch_shift(a.flatten(),res,i,j,b,stride,h,w,o_height,o_width,c,stream)
    res=res.reshape((b,o_height,o_width,c))
    return res

def test_vector_add():
    shape=(2,3,4) 
    
    a=torch.randn(shape)
    b=torch.randn(shape)
    c1=a+b
    print(c1)
    a=a.cuda()
    b=b.cuda()
    c=vector_add(a, b)
    print(c)

def test_bmm_1():
    batch_size=1
    m,n,p=66,129,256
    
    shape_a=(batch_size,m,n)
    shape_b=(batch_size,n,p)
    
    a=torch.randn(shape_a)
    b=torch.randn(shape_b)
    
    ref=torch.bmm(a,b).numpy()
    
    a=a.cuda()
    b=b.cuda()
    
    c=bmm(a, b)
    
    c=c.cpu().numpy()
    
    assert np.abs(c-ref).max()<1e-3




def test_bmm():
    for i in range(0,10):
        batch_size=np.random.randint(64,128)
        m,n,p=np.random.randint(512,1024,size=(3,))
        
        shape_a=(batch_size,m,n)
        shape_b=(batch_size,n,p)
        
        a=torch.randn(shape_a)
        b=torch.randn(shape_b)
        
        ref=torch.bmm(a,b).numpy()
        
        a=a.cuda()
        b=b.cuda()
        
        c=bmm(a, b)
        
        c=c.cpu().numpy()
        
        
        assert np.abs(c-ref).max()<1e-3
        info='test {num} passed, for shape {shape1} and {shape2}.\n The maximal absolute error is {e}.'.format(num=i,shape1=tuple(a.shape),
                                                                                                             shape2=tuple(b.shape),
                                                                                                               e=np.abs(c-ref).max())
        print(info)
    
def test_shift():
    x=torch.randn(size=(4,256,256,3)).numpy()
    w=torch.randn(size=(18,18,3,64)).numpy()
    stride=6
    padding=5
    
    batch_size=x.shape[0]
    in_channel=x.shape[-1]
    width=x.shape[2]
    height=x.shape[1]
    out_channel=w.shape[-1]
    
    o_height=(2*padding+height-w.shape[0])//stride+1
    o_width=(2*padding+width-w.shape[1])//stride+1
    
    x=np.pad(x, pad_width=((0,0),(padding,padding),(padding,padding),(0,0)))
    
    for i in range(0,w.shape[0]):
        for j in range(0,w.shape[1]):
            ref=x[:,i:i+o_height*stride:stride,j:j+o_width*stride:stride,:]
            out=shift(x, i, j, o_height, o_width, stride)
            assert np.abs(out-ref).max()<1e-3

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

def test_conv2d():
    pad=6
    st=5
    x=torch.randn(size=(5,256,256,3)).numpy()
    w=torch.randn(size=(32,32,3,64)).numpy()
    
    t1=time.time()
    out=conv2d(x, w, stride=st, padding=pad)  
    t2=time.time()
    ref=conv_np(x, w, stride=st, padding=pad)
    t3=time.time()
    
    acc=(t3-t2)/(t2-t1)
    print('maximal error:',np.abs(out-ref).max())
    print('acceleration:',acc)
    assert np.abs(out-ref).max()<1e-3


if __name__=="__main__":
    test_conv2d()
    
    