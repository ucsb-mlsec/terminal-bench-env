import numpy as np
import netCDF4 as nc

# Create the NetCDF file
ncfile = nc.Dataset('/data/pressure_grid.nc', 'w', format='NETCDF4')

# Define dimensions
time_dim = ncfile.createDimension('time', 48)
lat_dim = ncfile.createDimension('lat', 20)
lon_dim = ncfile.createDimension('lon', 30)

# Create coordinate variables
time_var = ncfile.createVariable('time', 'f4', ('time',))
lat_var = ncfile.createVariable('lat', 'f4', ('lat',))
lon_var = ncfile.createVariable('lon', 'f4', ('lon',))

# Create pressure variable
pressure_var = ncfile.createVariable('pressure', 'f4', ('time', 'lat', 'lon'))
pressure_var.units = 'hPa'
pressure_var.long_name = 'Atmospheric Pressure'

# Set coordinate values
time_var[:] = np.arange(48)
lat_var[:] = np.linspace(30, 45, 20)
lon_var[:] = np.linspace(-100, -85, 30)

# Initialize with valid pressure values (950-1050 hPa)
np.random.seed(42)
pressure_data = np.random.uniform(980, 1020, (48, 20, 30))

# Add realistic variations
for t in range(48):
    pressure_data[t] += np.random.uniform(-15, 15, (20, 30))

# Introduce data quality issues

# 1. Add missing values (NaN) - approximately 15% of data (~4320 points)
missing_indices = np.random.choice(28800, size=4320, replace=False)
for idx in missing_indices:
    t = idx // 600
    remainder = idx % 600
    i = remainder // 30
    j = remainder % 30
    pressure_data[t, i, j] = np.nan

# 2. Add invalid values - negative, zero, and >1200 hPa (~2000 points)
# Select points that are not already NaN
valid_mask = ~np.isnan(pressure_data.flatten())
valid_indices = np.where(valid_mask)[0]
invalid_selection = np.random.choice(valid_indices, size=2000, replace=False)

for idx in invalid_selection[:700]:
    t = idx // 600
    remainder = idx % 600
    i = remainder // 30
    j = remainder % 30
    pressure_data[t, i, j] = np.random.uniform(-150, -1)  # Negative values

for idx in invalid_selection[700:1000]:
    t = idx // 600
    remainder = idx % 600
    i = remainder // 30
    j = remainder % 30
    pressure_data[t, i, j] = 0.0  # Zero values

for idx in invalid_selection[1000:2000]:
    t = idx // 600
    remainder = idx % 600
    i = remainder // 30
    j = remainder % 30
    pressure_data[t, i, j] = np.random.uniform(1201, 2000)  # Above 1200 hPa

# 3. Create empty timesteps - make 3 complete timesteps empty
empty_timesteps = [5, 17, 35]
for t in empty_timesteps:
    # Mix of NaN and invalid values
    for i in range(20):
        for j in range(30):
            if np.random.random() < 0.6:
                pressure_data[t, i, j] = np.nan
            else:
                choice = np.random.choice(['negative', 'zero', 'high'])
                if choice == 'negative':
                    pressure_data[t, i, j] = -50.0
                elif choice == 'zero':
                    pressure_data[t, i, j] = 0.0
                else:
                    pressure_data[t, i, j] = 1500.0

# 4. Add stuck sensor values - some locations report same value repeatedly
stuck_locations = [(2, 5), (8, 15), (12, 22), (16, 8)]
for lat_idx, lon_idx in stuck_locations:
    stuck_value = np.random.uniform(995, 1015)
    for t in range(10, 25):  # Stuck for 15 hours
        if not np.isnan(pressure_data[t, lat_idx, lon_idx]):
            pressure_data[t, lat_idx, lon_idx] = stuck_value

# Write the pressure data
pressure_var[:] = pressure_data

# Add global attributes
ncfile.title = 'Regional Atmospheric Pressure Grid'
ncfile.institution = 'Weather Monitoring Network'
ncfile.source = 'Automated sensor array'
ncfile.history = 'Created with data quality issues for analysis'

# Close the file
ncfile.close()

print("NetCDF file created successfully at /data/pressure_grid.nc")