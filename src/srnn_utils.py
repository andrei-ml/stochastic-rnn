import torch
import numpy as np
import scipy
from torch.overrides import (
    handle_torch_function,
    has_torch_function,
    has_torch_function_unary,
    has_torch_function_variadic,
)


def mvgaussian_nll_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    cholesky_factor: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Compute the multivariate Gaussian negative log likelihood loss, inspired by torch.nn.functional.gaussian_nll_loss.

    Similar to torch.nn.functional.gaussian_nll_loss, but for multivariate Gaussian distribution. 
    To specify the covariance matrix, cholesky_factor parameter is used. It must be a square low-triangular 2-D Tensor (the values in upper-diagonal part will be ignored).
    

    Args:
        input: Expectation of the Gaussian distribution.
        target: Sample from the Gaussian distribution.
        cholesky_factor: Tensor (low-triangular matrix to specifying the covariance cholesky_factor @ cholesky_factor.T
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``
            Default: ``'mean'``.
    """
    if has_torch_function_variadic(input, target, cholesky_factor):
        return handle_torch_function(
            mvgaussian_nll_loss,
            (input, target, cholesky_factor),
            input,
            target,
            cholesky_factor,
            reduction=reduction,
        )

    # Check cholesky_factor size
    if cholesky_factor.dim() != 2 or cholesky_factor.size()[-1] != cholesky_factor.size()[-2] or cholesky_factor.size()[-1] != input.size()[-1]:
        raise ValueError("cholesky_factor is of incorrect size")

    # Check validity of reduction mode
    if reduction != "mean" and reduction != "sum":
        raise ValueError(reduction + " is not valid")

    # Calculate the loss factors
    #xi=torch.linalg.solve(torch.transpose(cholesky_factor,-1,-2),torch.reshape(target-input,(-1,input.shape[-1])),left=False)
    xi=torch.linalg.solve_triangular(torch.transpose(cholesky_factor,-1,-2),torch.reshape(target-input,(-1,input.shape[-1])),upper=True,left=False)
    loss1= 0.5 * (xi**2)
    loss2= 0.5 * torch.log(((torch.diagonal(cholesky_factor, offset=0, dim1=-2, dim2=-1))**2)*2. * np.pi)

    # Calculate the loss
    if reduction == "mean":
        return loss1.mean()+loss2.mean()
    else:
        return loss1.sum()+loss2.sum()
    
def mvgaussian_reduced_nll_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    return_cholesky_factor_instead_of_loss: bool = False,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Same as mvgaussian_reduced_nll_loss above, but minimized over cholesky_factor. This is done analyticaly.
    If return_cholesky_factor_instead_of_loss is True, then cholesky_factor is returned instead of loss. 

    Args:
        input: Expectation of the Gaussian distribution.
        target: Sample from the Gaussian distribution.
        return_cholesky_factor_instead_of_loss: bool
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``
            Default: ``'mean'``.
    """
    if has_torch_function_variadic(input, target):
        return handle_torch_function(
            mvgaussian_nll_loss,
            (input, target),
            input,
            target,
            return_cholesky_factor_instead_of_loss,
            reduction=reduction,
        )

    # Check validity of reduction mode
    if reduction != "mean" and reduction != "sum":
        raise ValueError(reduction + " is not valid")

    residuals=torch.reshape(target-input, (-1,target.shape[-1]))
    covariance_matrix=torch.mm(torch.transpose(residuals,0,1),residuals)/residuals.size(dim=0)
    cholesky_factor=torch.linalg.cholesky(covariance_matrix)
    if return_cholesky_factor_instead_of_loss:
        return cholesky_factor

    # Calculate the loss
    loss=0.5*(1.+np.log(2.*np.pi)+torch.mean(torch.log(torch.diagonal(cholesky_factor)**2)))
    #loss=0.5*(1.+torch.tensor(np.log(2.*np.pi),dtype=torch.get_default_dtype())+torch.mean(torch.log(torch.diagonal(cholesky_factor)**2)))
    
    if reduction == "mean":
        return loss
    else:
        return loss*torch.numel(target)
    
def scipy_minimize_bfgs(torch_function,torch_tensors,tol=0.,options={}):
    torch_tensors=list(torch_tensors)
    if (len(torch_tensors))==0: return 0    
    
    # Create starting point x0 as a flattened np.array for scipy.optimize.minimize
    np_arrays=[]
    sizes=[]
    shapes=[]
    for tensor in torch_tensors:
        np_array=tensor.detach().numpy()
        np_arrays.append(np_array.flatten())
        sizes.append(np_array.size)
        shapes.append(np_array.shape)
    x0=np.concatenate(np_arrays)
    
    # Function to put np.array content to the tensors, needed to compute torch_func and use autograd
    def update_torch_tensors(x):
        with torch.no_grad():
            ind=0
            i=0
            for i in range(len(torch_tensors)):
                torch_tensors[i].copy_(torch.from_numpy(np.reshape(x[ind:ind+sizes[i]],shapes[i])))
                ind+=sizes[i]
                i+=1

    def func(x):
        update_torch_tensors(x)
        with torch.no_grad(): f=torch_function().numpy()
        return f         

    def jac(x):
        update_torch_tensors(x)
        f=torch_function()
        with torch.no_grad(): gradients=torch.autograd.grad(f,torch_tensors)
        return np.concatenate([g.numpy().flatten() for g in gradients]) 
              
    res=scipy.optimize.minimize(func,x0,jac=jac,method='BFGS',tol=tol,options=options) 
    update_torch_tensors(res.x)  
    return res.nit

class InputsAndTargetsFromTimeSeries(torch.utils.data.Dataset):
    def __init__(self, input_ts, target_ts, lag, init_length, nfold, nfolds, mode):
        '''
        Returns input sequences of the shape [lag,dim_in] and target vectors [dim_out] which are used to predict data on the time interval [init_length:]
        All pairs sequences-targets are non-randomly splitted into equal groups with non-overlapping targets, consequently following each other. One of the groups returned as a test datta. 
        
        Parameters
        ----------
        input_ts : torch.Tensor[nsamples,N,dim_in]
        target_ts : torch.Tensor[nsamples,N,dim_out]
        lag : int>=1
        init_length: int>=lag
        nfold: number of test group
        nfolds: number of groups
        mode: 'train' or 'test', depending on which subset to return
        
        ----------

        '''
        self.lag=lag         
        #cutted and time-synchronised time series of intended input and target variables
        self.input_ts=input_ts[:,init_length-lag:,:]
        self.target_ts=target_ts[:,init_length-lag:,:]
        self.nsamples,self.N,self.dim_in=self.input_ts.shape
        
        if mode=='train':
            mask=np.full(nfolds,True)
            mask[nfold]=False
        elif mode=='test':
            mask=np.full(nfolds,False)
            mask[nfold]=True     
        
        #create two flat arrays of the shape [nsamples*(N-lag)] which store the index of sample and time for the target dataset
        isamples=np.repeat(np.arange(self.nsamples,dtype=int)[:,None],(self.N-self.lag),axis=-1).flatten()
        itimes=np.repeat(np.arange((self.N-self.lag),dtype=int)[None,:],self.nsamples,axis=0).flatten() 
        fold_size=(isamples.shape[0])//nfolds
        #the actual target data will be cutted so that its size is the highest multiple of nfolds
        #we return the part of the items which belongs to a target subgroup (train or test) 
        self.isamples=isamples[:fold_size*nfolds].reshape((nfolds,fold_size))[mask,:].flatten()
        self.itimes=itimes[:fold_size*nfolds].reshape((nfolds,fold_size))[mask,:].flatten()

    def __len__(self):
        return self.isamples.shape[0]

    def __getitem__(self, idx):
        # the function is suppossed to return the item with the number idx
        isample=self.isamples[idx]
        itime=self.itimes[idx]
        inputs=self.input_ts[isample,itime:itime+self.lag,:]
        targets=self.target_ts[isample,itime+self.lag:itime+self.lag+1,:]
        return inputs,targets