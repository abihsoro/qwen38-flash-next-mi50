# SPDX-License-Identifier: Apache-2.0
"""Opt-in gfx906 top-10 decode routing; float16 logits, 512 experts."""
import torch
import triton
import triton.language as tl
from triton.language.extra.hip import libdevice

@triton.jit
def _route(X,W,I,S,M:tl.constexpr,STRIDE:tl.constexpr,PAD,HAS_PAD:tl.constexpr):
    row=tl.program_id(0); cols=tl.arange(0,512)
    x=tl.load(X+row*STRIDE+cols).to(tl.float32)
    p=libdevice.exp(x-tl.max(x,0)); p=p/tl.sum(p,0)
    p=tl.where((p==p) & (tl.abs(p)!=float("inf")),p,0.0)
    j=tl.arange(0,16); vals=tl.full((16,),0,tl.float32); ids=tl.full((16,),0,tl.int32)
    for k in tl.static_range(10):
        best=tl.max(p,0)
        idx=tl.min(tl.where(p==best,cols,2147483647),0)
        vals=tl.where(j==k,best,vals); ids=tl.where(j==k,idx,ids)
        p=tl.where(cols==idx,-float('inf'),p)
    total=tl.sum(vals,0)
    vals=vals/tl.where(total>0,total,1.0)
    if HAS_PAD:
        ids=tl.where(tl.load(PAD+row),-1,ids)
    tl.store(W+row*10+j,vals,j<10); tl.store(I+row*10+j,ids,j<10)
    tl.store(S+row*10+j,j*M+row,j<10)

def candidate(x,warps=1,is_padding=None):
    m=x.shape[<bus>]
    w=torch.empty((m,10),device=x.device,dtype=torch.float32)
    i=torch.empty((m,10),device=x.device,dtype=torch.int32); s=torch.empty_like(i)
    _route[(m,)](x,w,i,s,m,x.stride(0),is_padding,is_padding is not None,num_warps=warps)
    return w,i,s
