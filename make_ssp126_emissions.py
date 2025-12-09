import numpy as np
import xarray as xr
import xarray as xr
import numpy as np

path="/scratch/project_462001112/emulator_data/"
ds_ssp126=xr.open_dataset(path+'emissions_ssp126.nc').sel(year=slice(2015,2100)).sortby("year")
ds =xr.open_dataset(path+'emissions.nc').sel(year=slice(1850,2014)).sortby("year")
ds_all = xr.concat([ds, ds_ssp126], dim="year")
print(ds_all)
ds_all.to_netcdf(path+"emissions_ssp126_v2.nc")



