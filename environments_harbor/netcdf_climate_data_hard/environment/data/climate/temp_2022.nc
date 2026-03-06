I'll generate the NetCDF file content for temp_2022.nc. Since this is a binary NetCDF file, I'll provide a Python script that creates it with the exact specifications:

import netCDF4 as nc
import numpy as np
import os

# Create directory if it doesn't exist
os.makedirs('/data/climate', exist_ok=True)

# Create NetCDF file
dataset = nc.Dataset('/data/climate/temp_2022.nc', 'w', format='NETCDF4')

# Define dimensions
time_dim = dataset.createDimension('time', 12)
lat_dim = dataset.createDimension('latitude', 21)
lon_dim = dataset.createDimension('longitude', 41)

# Create coordinate variables
times = dataset.createVariable('time', 'f8', ('time',))
latitudes = dataset.createVariable('latitude', 'f4', ('latitude',))
longitudes = dataset.createVariable('longitude', 'f4', ('longitude',))
temperature = dataset.createVariable('temperature', 'f4', ('time', 'latitude', 'longitude',))

# Add attributes
times.units = 'months since 2022-01-01'
times.calendar = 'gregorian'
times.long_name = 'time'

latitudes.units = 'degrees_north'
latitudes.long_name = 'latitude'

longitudes.units = 'degrees_east'
longitudes.long_name = 'longitude'

temperature.units = 'Kelvin'
temperature.long_name = 'Air Temperature'
temperature.standard_name = 'air_temperature'

# Populate coordinate data
times[:] = np.arange(0, 12)
latitudes[:] = np.arange(40.0, 50.5, 0.5)
longitudes[:] = np.arange(-10.0, 10.5, 0.5)

# Generate realistic temperature data for 2022
# Create base temperature pattern
lat_grid, lon_grid = np.meshgrid(latitudes[:], longitudes[:], indexing='ij')

# Monthly temperature cycle (seasonal variation) - 2022 pattern
# Slightly different from 2021, warmer spring, cooler autumn
monthly_base = np.array([275.5, 276.8, 280.2, 284.5, 289.3, 293.8, 
                         296.2, 295.5, 291.0, 285.2, 279.5, 276.0])

temp_data = np.zeros((12, 21, 41))

for month in range(12):
    # Base temperature for the month
    base_temp = monthly_base[month]
    
    # Latitude gradient (warmer in south, cooler in north)
    # About 0.6°C per degree latitude
    lat_effect = (50.0 - lat_grid) * 0.6
    
    # Longitude gradient (slight continental effect)
    lon_effect = lon_grid * 0.15
    
    # Add some spatial variability
    np.random.seed(month + 1000)  # Different seed than 2021
    noise = np.random.normal(0, 1.2, (21, 41))
    
    # Combine effects
    temp_data[month, :, :] = base_temp + lat_effect + lon_effect + noise
    
    # Add coastal cooling effect for western regions
    coastal_mask = lon_grid < -5
    temp_data[month, coastal_mask] -= 0.8
    
    # Add slight Mediterranean warming for southeast
    med_mask = (lat_grid < 43) & (lon_grid > 5)
    temp_data[month, med_mask] += 1.2

# Ensure temperatures are physically realistic
temp_data = np.clip(temp_data, 268.0, 305.0)

# Write temperature data
temperature[:, :, :] = temp_data

# Add global attributes
dataset.title = 'Monthly Temperature Data 2022'
dataset.institution = 'Meteorological Agency'
dataset.source = 'Climate Model Output'
dataset.history = 'Created for weather station reporting'
dataset.references = 'Internal climate archive'
dataset.comment = 'Gridded temperature data at 0.5 degree resolution'
dataset.year = 2022

# Close the dataset
dataset.close()

print("temp_2022.nc created successfully")