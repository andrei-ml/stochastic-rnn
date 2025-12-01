import os
os.environ["OMP_NUM_THREADS"] = "1"
# useful on linux to avoid numpy multithreading

import numpy as np, scipy, xarray as xr, joblib, sklearn
np.random.seed(1)

def old_load_and_prepare_data_v1(data_path, year1, year2):
    # Loading the dataset with a proper time decoder of new xarray
    ds=xr.open_mfdataset(os.path.join(data_path,'*.nc'), decode_times=xr.coding.times.CFDatetimeCoder(use_cftime=True))
    # Choose pressure level
    ds['zg']=ds['zg'].sel(plev=10000, method='nearest')
    ds=ds.drop_vars('plev')
    # Choose latitude range
    ds=ds.where(ds.lat >= 50, drop=True)
    ds=ds.where((ds.time.dt.year>=year1) & (ds.time.dt.year<=year2),drop=True)
    
    # Remove linear trend in time // not important here but can be used in general case
    fit = ds['zg'].polyfit(dim='time', deg=1, skipna=True)
    trend = xr.polyval(ds['zg'], fit.polyfit_coefficients)
    ds['zg'] = ds['zg'] - trend

    # Remove climatology
    anomalies,climatology=compute_seasonal_cycle_gaussian(ds['zg'], sigma=15)
    ds['zg']= anomalies
    
    # Cut and reshape to leave only N days per season    
    ds = ds.assign_coords(season_year=xr.where(
            ds.time.dt.month > 3, ds.time.dt.year+1, ds.time.dt.year)
                                                ) # call Apr-Mar a 'season_year' (=year number of Jan.-Mar.)
    ds=ds.where((ds.season_year>=year1+1) & (ds.season_year<=year2),drop=True) #remove incomplete seasons
    ds=ds.where(ds.time.dt.month.isin([9, 10, 11, 12, 1, 2, 3]), drop=True) # select only Sep-Mar
    N=212                                           # the number of Sep-Mar days in no-leap years
    season_years, season_sets = zip(*list(ds.groupby("season_year")))  # group and unzip into two sequences
    data = np.stack([s['zg'].values[:N,...] for s in season_sets],axis=0) #remove 31 Mar. in leap years + stack
    season_years=np.array(season_years)
    day_of_season_year=np.arange(N)+1

    # Return new 4D dataset
    ds_new = xr.Dataset(
    data_vars=dict(zg_a=(['season_year', 'day_of_season_year', 'lat', 'lon'], data),),
    coords=dict(season_year=season_years,day_of_season_year=day_of_season_year,lat=ds['lat'],lon=ds['lon']),
    attrs=dict(description='Prepared data')
    )
    ds.close()
    return ds_new

def old_compute_seasonal_cycle_gaussian(da, sigma=15):
    # Compute daily climatology including day 366
    # Note, dayofyear of the same date depends on the year (leap or no-leap)
    # Therefore, cimatology is based on the actual 'dayofyear' parameters, not the dates
    # e.g. if Dec. 31 is the day 365 or 366, clim. at day 366 will be underestimated
    # overall, this climatology makes sense up to 1-day shift back or forth
    # but the Gaussian filter further smoothes the effect
    clim_full = da.groupby("time.dayofyear").mean("time")

    # Temporarily drop da 366 for smoothing (if it exists in the data)
    clim_365 = clim_full.sel(dayofyear=clim_full.dayofyear != 366)

    # Extend climatology cyclically for smoothing
    clim_vals = clim_365.values
    clim_extended = np.concatenate([clim_vals, clim_vals, clim_vals], axis=0)
    # Apply Gaussian smoothing
    smoothed_extended = scipy.ndimage.gaussian_filter1d(clim_extended, sigma=sigma, axis=0, mode="wrap")
    # Remove extensions (keep middle year)
    smoothed_clim_365 = smoothed_extended[365:730,...]
    # Approximate day 366 manually
    day_366=((smoothed_clim_365[0,...]+smoothed_clim_365[-1,...])/2.)[None,...]
    smoothed_clim_full = np.concatenate([smoothed_clim_365, day_366] , axis=0)
    # Convert back to xarray.DataArray
    clim_full.values[:]= smoothed_clim_full

    # Subtract smoothed climatology from original data
    anomalies = da.groupby("time.dayofyear") - clim_full    
    return anomalies, clim_full

def load_and_prepare_data(data_path, year1, year2):
    # Loading the dataset with a proper time decoder of new xarray
    ds=xr.open_mfdataset(os.path.join(data_path,'*.nc'), decode_times=xr.coding.times.CFDatetimeCoder(use_cftime=True))
    # Choose pressure level
    ds['zg']=ds['zg'].sel(plev=10000, method='nearest')
    ds=ds.drop_vars('plev')
    # Choose latitude range
    ds=ds.where(ds.lat >= 50, drop=True)
    ds=ds.where((ds.time.dt.year>=year1) & (ds.time.dt.year<=year2),drop=True)
    
    # Convert to a list of full Apr-Mar 1-year intervals
    # call Apr-Mar a 'season_year' (=year number of Jan.-Mar.)
    ds = ds.assign_coords(season_year=xr.where(ds.time.dt.month > 3, ds.time.dt.year+1, ds.time.dt.year)) 
    ds=ds.where((ds.season_year>=year1+1) & (ds.season_year<=year2),drop=True) #remove incomplete seasons
    season_years, season_sets = zip(*list(ds.groupby("season_year")))  # group and unzip into two sequences

    # Remove 31. Mar in leap years + stack into 4-D array (along season_year axis)
    data = np.stack([s['zg'].values[:365,...] for s in season_sets],axis=0) 
    season_years=np.array(season_years)
    

    # Detrend the data
    nyears,N,nlat,nlon=data.shape
    data=scipy.signal.detrend(data.reshape(nyears*N,nlat,nlon), axis=0, type='linear')
    data=data.reshape(nyears,N,nlat,nlon)

    # Remove seasonal cycle using 15day Gaussian-filtered climatology
    clim=data.mean(axis=0)
    clim = np.concatenate([clim, clim, clim], axis=0)
    clim = scipy.ndimage.gaussian_filter1d(clim, sigma=15, axis=0, mode="wrap")
    clim = clim[N:2*N,...]
    data=data-clim

    # Cut September-Mar part of each 'season_year'
    data=data[:,-212:,:,:]
    day_of_season_year=np.arange(212)+1

    # Return as new 4D dataset
    ds_new = xr.Dataset(
    data_vars=dict(zg_a=(['season_year', 'day_of_season_year', 'lat', 'lon'], data),),
    coords=dict(season_year=season_years,day_of_season_year=day_of_season_year,lat=ds['lat'],lon=ds['lon']),
    attrs=dict(description='Prepared data')
    )
    ds.close()
    return ds_new

class CustomPreprocessor:
    def __init__(self,nan_mask,lat,n_pcs,n_kpcs,sigma):
        '''
        nan_mask -- mask of missing pixels [nlat,nlon]
        
        lat -- latitude coordinate, to produce weights
        
        sigma -- Gaussian kernel parameter (gamma=1/2/sigma**2)
        '''
        self.nan_mask=nan_mask
        self.lat=lat
        self.n_pcs=n_pcs
        self.n_kpcs=n_kpcs
        self.sigma=sigma
        self.gamma=1./2./(sigma)**2
        self.weights=np.sqrt(np.cos(np.deg2rad(lat))*np.sin(np.deg2rad(lat))**2)[:,None]

    def fit(self,data):
        '''
        data here must be 4-D as in the preparation step
        '''
        self.data_mean = data.mean(axis = (0,1))
        data=data-self.data_mean

        # applying specific weights
        data=data*self.weights

        # flattening the data along available grid nodes
        data = data[...,np.logical_not(self.nan_mask)]
        nyears,N,D=data.shape
        nlat,nlon=self.nan_mask.shape
        data=data.reshape((nyears*N,D))
        # Computing PCA
        self.pca=sklearn.decomposition.PCA(n_components=self.n_pcs)
        pcs=self.pca.fit_transform(data)
        pcs/=np.sqrt(self.pca.explained_variance_)

        # save EOFs ready to plot in physical units
        self.eofs=np.empty((self.n_pcs,nlat,nlon))
        self.eofs[:,self.nan_mask]=np.nan
        self.eofs[:,np.logical_not(self.nan_mask)]=self.pca.components_
        self.eofs=self.eofs*np.sqrt(self.pca.explained_variance_)[:,None,None]
        self.eofs=self.eofs/self.weights

        # Computing Kernel PCA
        norm=np.linalg.norm(data,axis=-1,keepdims=True)
        self.kpca = sklearn.decomposition.KernelPCA(n_components=self.n_kpcs, kernel='rbf', gamma=self.gamma)
        kpcs=self.kpca.fit_transform(data/norm)

        # Normalization parameters, for 
        self.pcs_scaler=sklearn.preprocessing.StandardScaler()
        self.pcs_scaler.fit(pcs)
        self.kpcs_scaler=sklearn.preprocessing.StandardScaler()
        self.kpcs_scaler.fit(kpcs) 

    def transform(self,data):
        data=data-self.data_mean

        # applying specific weights
        data=data*self.weights

        # flattening the data along available grid nodes
        data = data[...,np.logical_not(self.nan_mask)]
        nyears,N,D=data.shape
        data=data.reshape((nyears*N,D))
        
        # PCs (normalized to unit variance for the original data)
        pcs=self.pca.transform(data)
        pcs=pcs/np.sqrt(self.pca.explained_variance_)
        pcs=self.pcs_scaler.transform(pcs)
        pcs=pcs.reshape((nyears,N,self.n_pcs))

        # KPCs (normalized)
        norm=np.linalg.norm(data,axis=-1,keepdims=True)
        kpcs=self.kpca.transform(data/norm)
        kpcs=self.kpcs_scaler.transform(kpcs)
        kpcs=kpcs.reshape((nyears,N,self.n_kpcs))

        return pcs, kpcs

    def inverse_transform(self,pcs):
        # equivalent to np.tensordot(self.pcs,self.eofs,axes=(-1,0))+self.data_mean
        nyears,N,npcs=pcs.shape
        nlat,nlon=self.nan_mask.shape
        pcs=pcs*np.sqrt(self.pca.explained_variance_)
        pcs=pcs.reshape((nyears*N,self.n_pcs))
        data=self.pca.inverse_transform(pcs)
        data=data.reshape((nyears,N,-1))

        data_full=np.empty((nyears,N,nlat,nlon))
        data_full[...,self.nan_mask]=np.nan
        data_full[...,np.logical_not(self.nan_mask)]=data

        data_full=data_full/self.weights+self.data_mean

        return data_full

if __name__=='__main__':
    # Set path to the folder with the downloaded data files (should exist)
    data_path=os.path.expanduser('~/data/CMIP6.CMIP.MPI-M.MPI-ESM1-2-HR.piControl.r1i1p1f1.day.zg.gn')
    # And the path to the folder for the prepared data (should exist)
    prepared_data_path=os.path.expanduser('~/data/srnn_data/MPI-ESM')

    ds=load_and_prepare_data(data_path,2198,2273)
    evaluation_ds=load_and_prepare_data(data_path,2274,2349)
    
    ds.to_netcdf(os.path.join(prepared_data_path,'prepared_training_data.nc'))
    #ds=xr.open_dataset(os.path.join(prepared_data_path,'prepared_training_data.nc'))
    evaluation_ds.to_netcdf(os.path.join(prepared_data_path,'prepared_evaluation_data.nc'))
    #evaluation_ds=xr.open_dataset(os.path.join(prepared_data_path,'prepared_evaluation_data.nc'))
    
    data=ds['zg_a'].values

    # mask of missing pixels, here added for consistence
    nan_mask = np.isnan(data).any(axis=(0,1))

    # kernel parameter
    sigma=0.4 #1.35*0.3=0.405

    init_length=30+31 # Sep + Oct
    target_data=data[:,init_length:,:,:]

    # fit on target interval, but transform the full interval (with the memory for RNN)
    CP=CustomPreprocessor(nan_mask,lat=ds['lat'].values,n_pcs=100,n_kpcs=3,sigma=sigma)
    CP.fit(target_data)
    pcs, kpcs = CP.transform(data)
    evaluation_pcs, evaluation_kpcs = CP.transform(evaluation_ds['zg_a'].values)

    joblib.dump(CP,os.path.join(prepared_data_path,'CP.jpkl'))
    joblib.dump((pcs,kpcs,init_length),os.path.join(prepared_data_path,'training_data.jpkl'))
    joblib.dump((evaluation_pcs, evaluation_kpcs,init_length),os.path.join(prepared_data_path,'evaluation_data.jpkl'))


