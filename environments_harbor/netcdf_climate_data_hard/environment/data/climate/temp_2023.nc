from netCDF4 import Dataset
import numpy as np
import os

# Create output directory if it doesn't exist
os.makedirs('/data/climate', exist_ok=True)

# File path
filepath = '/data/climate/temp_2023.nc'

# Create NetCDF file
ncfile = Dataset(filepath, 'w', format='NETCDF4')

# Define dimensions
time_dim = ncfile.createDimension('time', 12)
lat_dim = ncfile.createDimension('latitude', 21)
lon_dim = ncfile.createDimension('longitude', 41)

# Create coordinate variables
times = ncfile.createVariable('time', 'f4', ('time',))
lats = ncfile.createVariable('latitude', 'f4', ('latitude',))
lons = ncfile.createVariable('longitude', 'f4', ('longitude',))
temp = ncfile.createVariable('temperature', 'f4', ('time', 'latitude', 'longitude',))

# Add attributes
times.units = 'months since 2023-01-01'
times.calendar = 'gregorian'
lats.units = 'degrees_north'
lons.units = 'degrees_east'
temp.units = 'Kelvin'
temp.long_name = 'Surface Air Temperature'

# Set coordinate values
times[:] = np.arange(0, 12)
lats[:] = np.arange(40.0, 50.5, 0.5)
lons[:] = np.arange(-10.0, 10.5, 0.5)

# Generate realistic temperature data for 2023
# Base temperatures by month (in Kelvin) - slightly warmer year
base_temps = np.array([276.5, 277.0, 280.5, 284.0, 288.5, 292.5, 
                       295.0, 294.5, 290.5, 285.5, 280.0, 277.5])

# Create temperature field with spatial and temporal variation
temp_data = np.zeros((12, 21, 41))

for month in range(12):
    for ilat in range(21):
        for ilon in range(41):
            lat_val = 40.0 + ilat * 0.5
            lon_val = -10.0 + ilon * 0.5
            
            # Base temperature for this month
            base = base_temps[month]
            
            # Latitude gradient (warmer in south, cooler in north)
            # About 0.5K per degree latitude
            lat_effect = -(lat_val - 45.0) * 0.5
            
            # Longitude effect (slight continental influence)
            # Warmer inland (positive longitude), cooler near Atlantic
            lon_effect = lon_val * 0.15
            
            # Add some spatial noise for realism
            noise = np.random.normal(0, 0.8)
            
            # Combine effects
            temp_data[month, ilat, ilon] = base + lat_effect + lon_effect + noise

# Ensure reasonable bounds (265K to 305K)
temp_data = np.clip(temp_data, 265.0, 305.0)

# Write temperature data
temp[:] = temp_data

# Add global attributes
ncfile.title = 'Monthly Temperature Data 2023'
ncfile.institution = 'Meteorological Agency'
ncfile.source = 'Climate Model Output'
ncfile.history = 'Created for weather station analysis'

# Close the file
ncfile.close()

print(f"NetCDF file created successfully at {filepath}")
print(f"Dimensions: time={12}, latitude={21}, longitude={41}")
print(f"Temperature range: {temp_data.min():.2f}K to {temp_data.max():.2f}K")