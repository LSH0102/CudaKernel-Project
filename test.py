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
    

if __name__=="__main__":
    
    test_bmm()
    
    