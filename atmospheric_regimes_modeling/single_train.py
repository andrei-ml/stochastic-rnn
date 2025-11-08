import os, joblib, copy
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
# useful on linux to avoid numpy multithreading
 
import numpy as np, scipy, torch
torch.set_default_dtype(torch.float64)
# can be omitted or changed to the preferred Tensors precision

# imports of the repository files, ensure that they work
try: 
    import srnn, srnn_utils
    # if the package was installed via pip command  
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), "src"))
    import srnn, srnn_utils
    # if the package is not installed into the environment, make sure the "path" variable below includes "src" folder containing srnn python files. Here -- "../src"


def single_train(input_ts,target_ts,options={'m': 2, 'lag': 1, 'nfold': 1, 'nfolds':5,'init_length': 61}):  
    '''
    Performs a single training and returns dictionary with the results
    '''
    #Specify the changeable parameters 
    m=options['m'] 
    # number of neurons to use in the RNN cell
    
    lag=options['lag'] 
    # the length of the RNN sequence 
    # (number of past values of input_ts involved in the prediction of the current value of the target_ts)
    
    init_length=options['init_length'] 
    # the number values at the start of time series which will not be used as a target 
    # (useful to keep the same training set and allow lag optimization <=init_length)
    
    dim_in, dim_out=input_ts.shape[-1],target_ts.shape[-1]
    nfolds=options['nfolds'] 
    # in cross-validation setting, it means that the data will be splitted into "nfolds" sequential parts, 
    # and one of these parts will be used for validation, and others are used in loss minimization
    
    nfold=options['nfold']
    # number of the validation fold, could be 0, 1, ..., nfolds-1         
    
    # save to the hyperparametr dictionary H
    H={'m':m,'lag':lag,'init_length':init_length,'dim_in':dim_in,'dim_out':dim_out,'nfolds':nfolds,'nfold':nfold}

    #to create the list of torch input/target pairs for SRNN model
    # 2) cut and split into train/validation subsets using a custom torch Dataset class in srnn_utils
    training_data=srnn_utils.InputsAndTargetsFromTimeSeries(
          input_ts,target_ts,lag,init_length,nfold=nfold,nfolds=nfolds,mode='train')
    train_dataloader=torch.utils.data.DataLoader(training_data, batch_size=len(training_data), shuffle=False)
    test_data=srnn_utils.InputsAndTargetsFromTimeSeries(
          input_ts,target_ts,lag,init_length,nfold=nfold,nfolds=nfolds,mode='test')
    test_dataloader=torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=False)  
    inputs, targets = list(train_dataloader)[0] 
    test_inputs, test_targets = list(test_dataloader)[0]
    #this creates train/test inputs ~ [N,lag,dim_in] and targets ~ [N,1,dim_out]  -- N pairs
    #could be alternatively done in a more explicit way without Dataloaders

    # create the model class, note that it is the same for any lag
    model=srnn.StochasticRNN(dim_in=dim_in,dim_out=dim_out,m=m)
    # set to the training mode
    model.train() 
    # fit using custom training function, with early stopping based on validation loss
    # BFGS scipy wrapper is used inside (usually efficient for low-dimensional setting)
    # The other option is "model.fit_adam" or make another custom training function
    training_status=model.fit_bfgs(inputs, targets,test_inputs, test_targets,options={})
    #training_status['opt_metrics'] contains most of necessary values, but we can compute them explicitly
    with torch.no_grad():
        training_loss=model.total_training_loss(inputs,targets).item()
        validation_loss=model.total_validation_loss(test_inputs,test_targets).item()
        weights=copy.deepcopy(model.state_dict()) 
    
    # if we don't want to save the whole model class, we ensure saving the weights an the hyperparameters H
    training_result={**H,**{'weights':weights,'training_loss':training_loss,'validation_loss':validation_loss}}
    return training_result