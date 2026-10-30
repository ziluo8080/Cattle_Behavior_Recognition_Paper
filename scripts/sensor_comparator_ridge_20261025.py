"""Same weighted Ridge objective, one column-major Gram allocation on GPU."""
import time
import numpy as np
import torch

def predict_scores(x,model,block=512):
    result=np.broadcast_to(model['intercept'],(len(x),len(model['classes']))).copy()
    for start in range(0,x.shape[1],block):result+=np.asarray(x[:,start:start+block],np.float64)@model['coef'][start:start+block]
    assert np.isfinite(result).all();return result

def predict(x,model):return model['classes'][predict_scores(x,model).argmax(1)]

def inplace_cholesky(matrix,rhs):
    assert matrix.stride()==(1,matrix.shape[0])
    info=torch.empty((),device=matrix.device,dtype=torch.int32)
    factor,info=torch.linalg.cholesky_ex(matrix,check_errors=False,out=(matrix,info))
    assert factor.data_ptr()==matrix.data_ptr() and int(info)==0
    # cholesky_solve may clone the entire n-by-n factor on CUDA. BLAS triangular
    # solves operate on the existing factor and allocate only n-by-7 outputs.
    intermediate=torch.linalg.solve_triangular(factor,rhs,upper=False)
    return torch.linalg.solve_triangular(factor.T,intermediate,upper=True)

def fit(x,y,weights,alpha=1.,block=512,progress=None,device='cuda'):
    progress=progress or (lambda **kwargs:None);started=time.perf_counter();n,d=x.shape
    classes=np.unique(y);assert len(classes)==7 and alpha==1.
    weights=np.asarray(weights,np.float64);assert np.isfinite(x).all() and np.isfinite(weights).all() and (weights>0).all()
    root=np.sqrt(weights);target=2*(y[:,None]==classes[None]).astype(np.float64)-1;target_mean=np.average(target,axis=0,weights=weights)
    rhs=torch.as_tensor(root[:,None]*(target-target_mean),device=device,dtype=torch.float64)
    gram=torch.zeros((n,n),device=device,dtype=torch.float64).T;means=np.empty(d,np.float64)
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    for start in range(0,d,block):
        values=np.asarray(x[:,start:start+block],np.float64);mean=np.average(values,axis=0,weights=weights);means[start:start+values.shape[1]]=mean
        z=torch.as_tensor(np.ascontiguousarray((values-mean)*root[:,None]),device=device,dtype=torch.float64)
        gram.addmm_(z,z.T);del z
        progress(stage='gram',columns=min(start+block,d),total_columns=d,seconds=time.perf_counter()-started)
    gram.diagonal().add_(alpha)
    if device=='cuda':torch.cuda.empty_cache()
    progress(stage='inplace_cholesky',samples=n)
    dual=inplace_cholesky(gram,rhs);del gram
    if device=='cuda':torch.cuda.empty_cache()
    coefficients=np.empty((d,7),np.float64);reconstructed=alpha*dual.clone()
    for start in range(0,d,block):
        values=np.asarray(x[:,start:start+block],np.float64)
        z=torch.as_tensor(np.ascontiguousarray((values-means[start:start+values.shape[1]])*root[:,None]),device=device,dtype=torch.float64)
        coef=z.T@dual;coefficients[start:start+values.shape[1]]=coef.cpu().numpy();reconstructed.addmm_(z,coef);del z,coef
    relative=float(torch.linalg.vector_norm(reconstructed-rhs)/torch.linalg.vector_norm(rhs))
    assert np.isfinite(relative) and relative<=1e-6,relative
    intercept=target_mean-means@coefficients;assert np.isfinite(coefficients).all() and np.isfinite(intercept).all()
    if device=='cuda':torch.cuda.synchronize()
    return dict(coef=coefficients,intercept=intercept,classes=classes,alpha=alpha,relative_system_residual=relative,fit_seconds=time.perf_counter()-started,
                peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,solver='float64 weighted centered dual Ridge, column-major in-place Cholesky, streamed original-system residual')

def witness(device='cpu'):
    rng=np.random.default_rng(17);results=[]
    for n,d in [(70,17),(70,160)]:
        x=rng.normal(size=(n,d))+2;y=np.arange(n)%7;w=rng.uniform(.25,4,n);query=rng.normal(size=(42,d))
        actual=fit(x,y,w,block=13,device=device)
        xm=np.average(x,axis=0,weights=w);target=2*(y[:,None]==np.arange(7)[None]).astype(float)-1;ym=np.average(target,axis=0,weights=w)
        xc=(x-xm)*np.sqrt(w)[:,None];yc=(target-ym)*np.sqrt(w)[:,None]
        coef=np.linalg.solve(xc.T@xc+np.eye(d),xc.T@yc);expected=(query-xm)@coef+ym
        error=float(np.max(np.abs(predict_scores(query,actual)-expected)))
        assert error<1e-6;np.testing.assert_array_equal(predict(query,actual),expected.argmax(1))
        results.append(dict(n=n,d=d,max_score_error=error,residual=actual['relative_system_residual']))
    return results

def full_shape_witness(n,columns=256):
    torch.manual_seed(17);torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    x=torch.randn(n,columns,device='cuda',dtype=torch.float64);rhs=torch.randn(n,7,device='cuda',dtype=torch.float64)
    matrix=torch.empty((n,n),device='cuda',dtype=torch.float64).T;torch.mm(x,x.T,out=matrix);matrix.diagonal().add_(1.)
    torch.cuda.synchronize();gram_seconds=time.perf_counter()-start;start=time.perf_counter();dual=inplace_cholesky(matrix,rhs)
    torch.cuda.synchronize();solve_seconds=time.perf_counter()-start;relative=float(torch.linalg.vector_norm(x@(x.T@dual)+dual-rhs)/torch.linalg.vector_norm(rhs))
    assert relative<=1e-6
    result=dict(n=n,columns=columns,gram_seconds=gram_seconds,solve_seconds=solve_seconds,residual=relative,peak_cuda_bytes=torch.cuda.max_memory_allocated())
    del matrix,x,rhs,dual;torch.cuda.empty_cache();return result
