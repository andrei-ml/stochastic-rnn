import torch
import numpy as np
from copy import deepcopy
import srnn_utils

class RNN_core(torch.nn.Module):
    '''
    Custom RNN parameterization without stochastic part. 
    It applies RNN cell to a sequence of an arbitrary length, starting with the latent state h=0 at the start of the sequence. 
    Then, a linear layer is applied to the latent state h at the end of the sequence. 
    
    Parameters
    ----------
    dim_in : int>0, input dimension
    dim_out: int>0, output dimension    
    m : int>0, number of neurons (dimension of h)
    bias: bool, whether to apply bias to output layer

    Methods:
        forward(x): 

                x[nsamples,lag,dim_in] -> f[nsamples,1,dim_out]

    '''      
    def __init__(self,dim_in,dim_out,m,bias=False):             
        super().__init__()   
        self.m=m
        self.dim_in=dim_in
        self.dim_out=dim_out
        self.rnn=torch.nn.RNN(input_size=dim_in,hidden_size=m,num_layers=1,nonlinearity='tanh',batch_first=False)
        self.linear=torch.nn.Linear(in_features=m, out_features=dim_out, bias=bias)

        for pp in self.parameters():
            torch.nn.init.trunc_normal_(pp) 

        self.output_scaling=1./np.sqrt((max(1,self.m)))
        self.input_scaling=1./np.sqrt((max(1,self.dim_in)))   

    def forward(self, x):
        '''
        x[nsamples,lag,dim_in]

        Outputs:
        f: [nsamples,1,dim_out]
        '''
        x=torch.swapaxes(x, 0,1)
        #x[lag,nsamples,d]
        
        output,h=self.rnn(x*self.input_scaling)
        h=h[0,:,:]
        out=self.linear(h)[:,None,:]
        return out*self.output_scaling
        
class StochasticRNN_core(torch.nn.Module):
    '''
    Stochastic RNN parameterization. Forward pass computes and returns the RNN_core term f(x) and the matrix sigma, 
    so that the full model prediction is drawn from normal distribution with the mean 'f(x)' and covariance matrix 'sigma @ sigma.T', or, equivalently:
        
        stochastic_prediction(x)=f(x)+sigma @ gaussian_standard_random_variable_vector

    '''
    def __init__(self,dim_in,dim_out,m,bias=False):

        super().__init__() 
        self.m=m
        self.dim_in=dim_in
        self.dim_out=dim_out
        self.f = RNN_core(dim_in,dim_out,m)  
        self.sigma = torch.nn.Parameter(torch.zeros(size=(dim_out,dim_out)))
        torch.nn.init.trunc_normal_(self.sigma) 
        
    def forward(self, x):
        '''
        x[nsamples,lag,d]
          
        Outputs:
        f: [nsamples,1,d]
        g: [d,d]      
        '''
        f=self.f(x)
        sigma=torch.tril(self.sigma)
        return f, sigma
    
    def predict(self, x):
        nsamples,lag,d=x.size()
        f,sigma=self.forward(x)
        noise=torch.tensor(np.random.normal(size=(nsamples,self.dim_out)))
        y=f+torch.matmul(sigma,noise[:,:,None])[:,None,:,0]
        return y

class StochasticRNN(StochasticRNN_core):
    '''
    Stochastic RNN parameterization inherited from StochasticRNN_core, with the loss and various fit functions added. 
    Fitting is held analytically over sigma parameter, which is assumed to be lower-triangular matrix.
    Other variants of sigma parameterization are possible but require at least the change of fit_sigma, base_loss_reduced, total_training_loss_reduced methods.

    Regularization is implemented assuming that the inputs and targets time series are all normalized to have mean ~0 and variance ~1.
    '''
    def __init__(self,dim_in,dim_out,m):
        super().__init__(dim_in,dim_out,m)         

    def base_loss(self,inputs,targets):
        '''
        Compute loss function term before applying regularization terms.
        Here -- the multivariate Gaussian negative log-likelihood loss for model predictions, divided by the number of target values
        Uses only lower-triangular part of sigma parameter after forward pass.
        Intended only for non-minimization purposes. Explicit minimization of it could cause unstable behavior if sigma @ sigma.T is not ensured to be positive-definite for all possible entries (i.e. zeroes excluded)

        :inputs: Input data for the forward pass
        :targets: Target values for loss computation
        :return: The computed base loss
        '''   
        f_out,sigma_out = self.forward(inputs)
        #print(CrossGaussianNLL_fn(f_out, sigma_out,targets)/torch.numel(targets),srnn_utils.mvgaussian_nll_loss(f_out,targets,sigma_out))
        #return CrossGaussianNLL_fn(f_out, sigma_out,targets)/torch.numel(targets)
        return srnn_utils.mvgaussian_nll_loss(f_out,targets,sigma_out)
    
    def base_loss_reduced(self,inputs,targets): 
        '''
        Compute reduced loss function term before applying regularization terms.
        Here -- the multivariate Gaussian negative log-likelihood loss for model predictions, divided by the number of target values.
        "Reduced" means that the loss in already optimized over sigma parameter.
        Here it is done assuming that sigma is lower-triangular matrix parameter (cholesky_factor, see srnn_utils.mvgaussian_reduced_nll_loss).

        :inputs: Input data for the forward pass
        :targets: Target values for loss computation
        :return: The computed base loss
        '''             
        f_out,sigma_out = self.forward(inputs)
        #return CrossGaussianNLLreduced_fn(f_out,targets,extend=False)/torch.numel(targets)
        return srnn_utils.mvgaussian_reduced_nll_loss(f_out,targets)

    def fit_sigma(self,inputs,targets):
        '''
        For a specific base_loss and uninformative regularization for sigma, we solve minimization over sigma analytically.
        Here it is done assuming that sigma is lower-triangular matrix parameter.
        '''
        with torch.no_grad():
            f_out,sigma_out = self.forward(inputs)
            sigma=srnn_utils.mvgaussian_reduced_nll_loss(f_out,targets,return_cholesky_factor_instead_of_loss=True)
            # Remember that only the lower-triangular part of sigma parameter will be used in the RNN forward pass
            self.sigma.copy_(sigma)             

    def reg_loss(self):
        '''
        Regularization term corresponding to the Gaussian negative prior probability density function for all weights, with zero mean and unit standard deviation.
        '''
        ret=0.
        for p in self.f.parameters():
            ret=ret+0.5*torch.sum((p**2)+np.log(2.*np.pi))
        return ret

    def total_training_loss(self,inputs,targets):
        '''
        total_training_loss=base_loss+reg_loss/number_of_target_values
        '''
        return self.base_loss(inputs,targets)+self.reg_loss()/torch.numel(targets)
    
    def total_training_loss_reduced(self,inputs,targets):
        '''
        total_training_loss_reduced=base_loss_reduced+reg_loss/number_of_target_values
        Like the base_loss, intended only for non-minimization purposes
        '''
        return self.base_loss_reduced(inputs,targets)+self.reg_loss()/torch.numel(targets)
    
    def total_validation_loss(self,test_inputs,test_targets):
        '''
        total_validation_loss=base_loss
        '''
        return self.base_loss(test_inputs,test_targets)    

    def compute_metrics(self,inputs,targets,test_inputs,test_targets):
        metrics={}
        with torch.no_grad():
            metrics['training_base_loss']=self.base_loss(inputs,targets).item()
            metrics['training_loss'] = self.total_training_loss(inputs,targets).item()
            metrics['validation_loss']=self.total_validation_loss(test_inputs,test_targets).item()
            f_out,sigma_out = self.forward(inputs)
            metrics['training_mse_loss']=torch.nn.functional.mse_loss(f_out,targets).numpy()
        return metrics

    def _update_optimum(self,inputs,targets,test_inputs,test_targets,training_status=None,new_training_loss=None,other={}):        
        # Extract numpy value from new_training_loss
        with torch.no_grad():
            if new_training_loss is None:
                new_training_loss=self.total_training_loss(inputs,targets).item()
            else:
                new_training_loss=new_training_loss.item()

        # Check if we need to update anything (either the 1st iteration or the min_training_loss is improved)
        if training_status is None: training_status={}
        if ('min_training_loss' not in training_status.keys()) or new_training_loss<=training_status['min_training_loss']:
            # If yes, record it into history 
            with torch.no_grad():
                new_validation_loss=self.total_validation_loss(test_inputs,test_targets).item()
            if 'history' not in training_status.keys(): training_status['history']=[]
            training_status['history'].append({**other,**{'training_loss':new_training_loss,'validation_loss':new_validation_loss}})
            # Update the minimum taining loss
            training_status['min_training_loss']=new_training_loss
            # Update the optimum if the validation loss is improved
            if ('min_validation_loss' not in training_status.keys()) or new_validation_loss<training_status['min_validation_loss']:
                training_status['opt_weights']=deepcopy(self.state_dict())
                training_status['min_validation_loss']=new_validation_loss
                training_status['no_change_counter']=0 # Either create or reset the counter of unsuccessful checks
            else:
                training_status['no_change_counter']+=1 # increment the counter
        return training_status
        
    def _fit_preliminary_adam(self,inputs, targets,test_inputs, test_targets,options={}):
        options={**{'pre_lr1':0.1,'pre_lr2':0.01,'pre_max_iter':100}, **options}
        max_iter1=options['pre_max_iter']
        lr1=options['pre_lr1']
        lr2=options['pre_lr2']
        optimizer=torch.optim.Adam(self.f.parameters(),lr=lr1)
        optimizer.zero_grad()
        #Slow exponential decay
        scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=np.exp(np.log(lr2/lr1)/max_iter1))    

        #record initial state
        with torch.no_grad():
            loss = self.total_training_loss(inputs,targets)
            iteration_count=0
            training_status=self._update_optimum(inputs,targets,test_inputs,test_targets,training_status=None,new_training_loss=loss,
                                                     other={'iteration_count':iteration_count,'lr':optimizer.param_groups[0]['lr']})                       

        #update with initial state with optimized sigma
        optimizer.zero_grad()
        loss=self.total_training_loss_reduced(inputs,targets)
        self.fit_sigma(inputs,targets)
        iteration_count=1
        training_status=self._update_optimum(inputs,targets,test_inputs,test_targets,training_status,new_training_loss=loss,
                                                     other={'iteration_count':iteration_count,'lr':optimizer.param_groups[0]['lr']})         
    
        #perform optimization   
        while(iteration_count-1<max_iter1):
            loss.backward()
            optimizer.step() # step by f
            scheduler1.step()
            optimizer.zero_grad()
            loss=self.total_training_loss_reduced(inputs,targets)
            self.fit_sigma(inputs,targets) # step by sigma
            iteration_count+=1
            training_status=self._update_optimum(inputs,targets,test_inputs,test_targets,training_status,new_training_loss=loss,
                                                     other={'iteration_count':iteration_count,'lr':optimizer.param_groups[0]['lr']})
            
            #        training_curve.append([iteration_count,min_loss,min_test_NLL,min_train_NLL,optimizer.param_groups[0]['lr']])
                    
        self.load_state_dict(training_status['opt_weights'])
        training_status.pop('opt_weights') # No need to keep large tensors, it's in the model class
        training_status['opt_metrics']=self.compute_metrics(inputs,targets,test_inputs,test_targets)
        training_status['opt_metrics']['iteration_count']=iteration_count
        
        return training_status

    def fit_bfgs(self,inputs, targets,test_inputs, test_targets,options={},pre_fit_with_adam=True):
        options={**{'max_iter_bfgs':20,'no_improve_count_max':3,'pre_lr1':0.1,'pre_lr2':0.01,'pre_max_iter':100},**options}
        # call the other function before fitting if specified
        pre_fit_training_status_history=None
        if pre_fit_with_adam: pre_fit_training_status_history=self._fit_preliminary_adam(inputs, targets,test_inputs, test_targets,options)['history']

        #print(inputs.sum(),targets.sum(),test_inputs.sum(),test_targets.sum())  
        #train_config=par#{'max_iter1':20000,'max_iter2':2000,'lr1':0.1,'lr2':1e-5,'improve_threshold':1e-6}
        max_iter_bfgs=options['max_iter_bfgs']  
        no_improve_count_max=options['no_improve_count_max']
        
        #record initial state
        with torch.no_grad():
            loss = self.total_training_loss(inputs,targets)
            iteration_count=0
            training_status=self._update_optimum(inputs,targets,test_inputs,test_targets,training_status=None,new_training_loss=loss,
                                                     other={'iteration_count':iteration_count})  
            

        # record pre_fit history
        training_status['pre_fit_history']=pre_fit_training_status_history

        #update with initial state with optimized sigma
        loss=self.total_training_loss_reduced(inputs,targets)
        self.fit_sigma(inputs,targets)
        iteration_count=1
        training_status=self._update_optimum(inputs,targets,test_inputs,test_targets,training_status,new_training_loss=loss,
                                                     other={'iteration_count':iteration_count})
        
        
        #perform optimization   
        def closure():
            loss=self.total_training_loss_reduced(inputs,targets)
            return loss    
        parameters=list(self.f.parameters())

        flag=True
        curr_nit=1   
        nit=1
        while (nit>0) and flag:
            ## Make curr_nit steps and check if the validation loss has decreased
            i=0
            while (i<curr_nit) and (nit>0) and (flag):
                nit=srnn_utils.scipy_minimize_bfgs(closure,parameters,options={'maxiter':min(100,curr_nit),'gtol': 1e-20, 'disp': False})
                loss=self.total_training_loss_reduced(inputs,targets)
                self.fit_sigma(inputs,targets)
                iteration_count+=nit
                training_status=self._update_optimum(inputs,targets,test_inputs,test_targets,training_status,new_training_loss=loss,
                                                     other={'iteration_count':iteration_count})
                if training_status['no_change_counter']>=no_improve_count_max:
                    flag=False
                i+=nit
            curr_nit=min(curr_nit*2,max_iter_bfgs)
                    
        self.load_state_dict(training_status['opt_weights'])
        training_status.pop('opt_weights') # No need to keep large tensors, it's in the model class
        training_status['opt_metrics']=self.compute_metrics(inputs,targets,test_inputs,test_targets)
        training_status['opt_metrics']['iteration_count']=iteration_count
        
        return training_status
    
    def fit_adam(self,inputs, targets,test_inputs, test_targets,options={}):
        options={**{'max_iter1':20000,'max_iter2':2000,'lr1':0.1,'lr2':1e-5,'improve_threshold':1e-6},**options}
        max_iter1=options['max_iter1']
        max_iter2=options['max_iter2']
        improve_threshold=options['improve_threshold']
        nit=500
        lr1=options['lr1']
        lr2=options['lr2']
        optimizer=torch.optim.Adam(self.f.parameters(),lr=lr1)
        
        #Slow exponential decay
        scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=np.exp(np.log(lr2/lr1)/max_iter1))    
        
        #Fast decay if loss decreasing stagnates
        scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',eps=0.,patience=0,cooldown=0,
                                                                threshold=improve_threshold,threshold_mode='abs',
                                                                factor=np.exp(np.log(lr2/lr1)/max_iter2))

        #Momentary decay to lr below lr2 if optimal test_NLL is not significantly updated after 'patience' min_loss updates
        scheduler3 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',eps=0.,patience=2000,cooldown=0,
                                                                threshold=improve_threshold,threshold_mode='abs',
                                                                factor=0.1*lr2/lr1)
          
        #record initial state
        with torch.no_grad():
            loss = self.total_training_loss(inputs,targets)
            iteration_count=0
            training_status=self._update_optimum(inputs,targets,test_inputs,test_targets,training_status=None,new_training_loss=loss,
                                                     other={'iteration_count':iteration_count,'lr':optimizer.param_groups[0]['lr']})  
            

        #update with initial state with optimized sigma
        optimizer.zero_grad()
        loss=self.total_training_loss_reduced(inputs,targets)
        self.fit_sigma(inputs,targets)
        iteration_count=1
        training_status=self._update_optimum(inputs,targets,test_inputs,test_targets,training_status,loss,
                                                     other={'iteration_count':iteration_count,'lr':optimizer.param_groups[0]['lr']})
        
    
        #optimization   
        curr_nit=5   
        while(optimizer.param_groups[0]['lr']>=lr2):
            ## Make curr_nit steps and check if the validation loss has decreased
            i=0
            while (i<curr_nit) and (optimizer.param_groups[0]['lr']>=lr2):
                loss.backward()
                optimizer.step()
                scheduler1.step()
                scheduler2.step(loss)
                scheduler3.step(training_status['history'][-1]['validation_loss']) # tracking last recorded validation loss in this scheduler 

                optimizer.zero_grad()
                loss=self.total_training_loss_reduced(inputs,targets)
                self.fit_sigma(inputs,targets)
                iteration_count+=1
                training_status=self._update_optimum(inputs,targets,test_inputs,test_targets,training_status,loss,
                                                     other={'iteration_count':iteration_count,'lr':optimizer.param_groups[0]['lr']})                
                i+=1
            
            curr_nit=min(curr_nit*5,nit)
                    
        self.load_state_dict(training_status['opt_weights'])
        training_status.pop('opt_weights') # No need to keep large tensors, it's in the model class
        training_status['opt_metrics']=self.compute_metrics(inputs,targets,test_inputs,test_targets)
        training_status['opt_metrics']['iteration_count']=iteration_count
        
        return training_status