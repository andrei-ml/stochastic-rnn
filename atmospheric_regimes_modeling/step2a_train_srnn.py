import sys, os, joblib, copy, multiprocessing, pickle
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
# useful on linux to avoid numpy multithreading

import numpy as np, scipy, torch
torch.set_default_dtype(torch.float64)
# can be omitted or changed to the preferred precision
 
from single_train import single_train

# Note, this is the custom way of preparing inputs and ouputs. Changing this part should be synchronized with the changes at step4 (predict_time_series function)
def make_ts_of_inputs_and_targets(pcs,kpcs):
    input_ts=torch.tensor(pcs[...,:3],dtype=torch.get_default_dtype()) 
    target_ts=torch.tensor(np.concatenate([pcs[...,:3],kpcs[...,:3]],axis=-1),dtype=torch.get_default_dtype())    
    return input_ts, target_ts

if __name__=='__main__':
    n_jobs=int(sys.argv[1])
    prepared_data_path=os.path.expanduser('~/data/srnn_data/MPI-ESM')
    trained_srnn_parameters_file=os.path.join(prepared_data_path,'trained_srnn_parameters_2a.pkl')

    # define option list to parallel
    options_list=[]
    nfolds=5
    for n in range(10):
        for m in range(1,25+1):
            for lag in list(range(1,28+1)):
                for nfold in range(nfolds):
                    options={'n':n,'m': m, 'lag': lag, 'nfold': nfold, 'nfolds':nfolds}
                    if options not in options_list: options_list.append(options)
    # We can add as many loops as we want. The script will not estimate what already was estimated after re-launching.
    # E.g. new loop can be added after looking at the optimal results at step 3
    # In this way we can keep all the iterations made to allow reproducing the same grid search scheme
    for n in range(30):
        for m in range(10,20+1):
            for lag in [8,10,12,13,14,15,16,17,18,21,28]:
                for nfold in range(nfolds):
                    options={'n':n,'m': m, 'lag': lag, 'nfold': nfold, 'nfolds':nfolds}
                    if options not in options_list: options_list.append(options)
    for n in range(100):
        for m in [9,10,11,12,13,14,15]:
            for lag in range(1,20):                
                for nfold in range(nfolds):
                    options={'n':n,'m': m, 'lag': lag, 'nfold': nfold, 'nfolds':nfolds}
                    if options not in options_list: options_list.append(options)
    for n in range(200):
        for m in [11,12,13,14,15]:
            for lag in [6,7,8,9,10,11,12,13]:                
                for nfold in range(nfolds):
                    options={'n':n,'m': m, 'lag': lag, 'nfold': nfold, 'nfolds':nfolds}
                    if options not in options_list: options_list.append(options)

    for n in range(300):
        for m in [11,12,13,14,15]:
            for lag in [8,9,10,11,12,13]:                
                for nfold in range(nfolds):
                    options={'n':n,'m': m, 'lag': lag, 'nfold': nfold, 'nfolds':nfolds}
                    if options not in options_list: options_list.append(options)

    # Read the previously existing results to avoid repeating them
    # The storage format is a single pickle file with multiple consecutive dumps
    # It is chosen to avoid multiple files and reloading of the full file at each iteration 
    my_list = []
    if os.path.isfile(trained_srnn_parameters_file):
        with open(trained_srnn_parameters_file, 'rb') as f:    
            while True:
                try:
                    d=pickle.load(f)
                    d = {k: d[k] for k in options_list[0].keys()}
                    my_list.append(d)
                except EOFError:
                    break

    # Remove the tasks which are already done
    options_list = [d for d in options_list if d not in my_list]

    # Create a lock shared by all jobs, to safely write to the file
    lock = multiprocessing.Lock()

    def single_job(options, trained_srnn_parameters_file):      
        # load the training data
        pcs,kpcs,init_length=joblib.load(os.path.join(prepared_data_path,'training_data.jpkl'))
        input_ts, target_ts=make_ts_of_inputs_and_targets(pcs,kpcs)        
        # train
        options={**options, **{'init_length':init_length}}
        trained_rnn_parameters=single_train(input_ts,target_ts,options)        
        # save
        trained_rnn_parameters['n']=options['n']
        with lock:
            with open(trained_srnn_parameters_file,'ab') as f:
                pickle.dump(trained_rnn_parameters,f)
        #print something, because it may take large time
        print({key:trained_rnn_parameters[key] for key in ['n','m','lag','nfold','training_loss','validation_loss']}, flush=True)
            
    joblib.Parallel(n_jobs=n_jobs,backend="multiprocessing")(
        joblib.delayed(single_job)(options, trained_srnn_parameters_file)
        for options in options_list
    )
    
    print('Finish',flush=True)



