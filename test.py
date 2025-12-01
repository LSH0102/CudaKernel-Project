# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 18:49:00 2025

@author: 25488
"""
import ctypes
import torch
import numpy as np
lib_add=ctypes.CDLL("cuda_kernels/kernel.so")
datatype=np.float32

def vector_add(a:torch.Tensor,b:torch.Tensor):
    size=a.numel()
    shape=a.shape
    c=torch.zeros_like(a).view(-1)
    stream=torch.cuda.current_stream().cuda_stream
    
    
    lib_add.launch_add.argtypes=[
        
        np.ctypeslib.ndpointer(dtype=datatype,ndim=1,flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype,ndim=1,flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype,ndim=1,flags='C_CONTIGUOUS'),
        ctypes.c_int,
        ctypes.c_void_p]
    
    lib_add.launch_add.restype=None
    
    res=c.view(-1).cpu().numpy()
    lib_add.launch_add(a.view(-1).cpu().numpy(),b.view(-1).cpu().numpy(),res,size,stream)
    torch.cuda.synchronize()
    
    c=torch.from_numpy(res).to(a.device)
    c=c.reshape(shape)
    return c


if __name__=="__main__":
    
    shape=(2,3,4) 
    
    a=torch.randn(shape)
    b=torch.randn(shape)
    c1=a+b
    print(c1)
    a=a.cuda()
    b=b.cuda()
    c=vector_add(a, b)
    print(c)
    
    