I'll generate the NetCDF file for temp_2021.nc with realistic European climate data.

```python
import numpy as np
from netCDF4 import Dataset
import os

# Create directory if it doesn't exist
os.makedirs('/data/climate', exist_ok=True)

# Create NetCDF file
nc_file = Dataset('/data/climate/temp_2021.nc', 'w', format='NETCDF4')

# Define dimensions
time_dim = nc_file.createDimension('time', 12)
lat_dim = nc_file.createDimension('latitude', 21)
lon_dim = nc_file.createDimension('longitude', 41)

# Create coordinate variables
time_var = nc_file.createVariable('time', 'f4', ('time',))
lat_var = nc_file.createVariable('latitude', 'f4', ('latitude',))
lon_var = nc_file.createVariable('longitude', 'f4', ('longitude',))
temp_var = nc_file.createVariable('temperature', 'f4', ('time', 'latitude', 'longitude'))

# Set coordinate values
time_var[:] = np.arange(0, 12)  # months 0-11
lat_var[:] = np.arange(40.0, 50.5, 0.5)  # 40.0 to 50.0 in 0.5° increments
lon_var[:] = np.arange(-10.0, 10.5, 0.5)  # -10.0 to 10.0 in 0.5° increments

# Add attributes
time_var.units = 'months since 2021-01-01'
time_var.calendar = 'gregorian'
time_var.long_name = 'time'

lat_var.units = 'degrees_north'
lat_var.long_name = 'latitude'

lon_var.units = 'degrees_east'
lon_var.long_name = 'longitude'

temp_var.units = 'Kelvin'
temp_var.long_name = 'air_temperature'

# Generate realistic temperature data
# Base temperature pattern: warmer in south, cooler in north
# Seasonal variation: cold in winter (Jan, Feb, Dec), warm in summer (Jun, Jul, Aug)

for month in range(12):
    # Seasonal temperature component (Kelvin)
    # January (0) = cold, July (6) = warm
    seasonal_factor = -12 * np.cos(2 * np.pi * month / 12)  # Ranges from -12K (winter) to +12K (summer)
    
    for i, lat in enumerate(lat_var[:]):
        for j, lon in enumerate(lon_var[:]):
            # Base temperature: warmer in south (41°N warmer than 50°N)
            base_temp = 286.15 - (lat - 40.0) * 0.5  # Decreases with latitude
            
            # Add seasonal variation
            temp = base_temp + seasonal_factor
            
            # Add some spatial variation based on longitude (maritime vs continental)
            maritime_effect = 2.0 * np.cos(np.radians(lon * 2))
            temp += maritime_effect
            
            # Add small random variation
            np.random.seed(2021 * 100 + month * 1000 + i * 10 + j)
            noise = np.random.normal(0, 1.5)
            temp += noise
            
            temp_var[month, i, j] = temp

# Add global attributes
nc_file.title = 'Monthly Temperature Data 2021'
nc_file.institution = 'Meteorological Agency'
nc_file.source = 'Climate Model'
nc_file.history = 'Created for weather station reporting'
nc_file.references = 'Internal archive'
nc_file.comment = 'Gridded temperature data at 0.5 degree resolution'

# Close file
nc_file.close()

print("temp_2021.nc created successfully")

# Create temp_2022.nc
nc_file = Dataset('/data/climate/temp_2022.nc', 'w', format='NETCDF4')

time_dim = nc_file.createDimension('time', 12)
lat_dim = nc_file.createDimension('latitude', 21)
lon_dim = nc_file.createDimension('longitude', 41)

time_var = nc_file.createVariable('time', 'f4', ('time',))
lat_var = nc_file.createVariable('latitude', 'f4', ('latitude',))
lon_var = nc_file.createVariable('longitude', 'f4', ('longitude',))
temp_var = nc_file.createVariable('temperature', 'f4', ('time', 'latitude', 'longitude'))

time_var[:] = np.arange(0, 12)
lat_var[:] = np.arange(40.0, 50.5, 0.5)
lon_var[:] = np.arange(-10.0, 10.5, 0.5)

time_var.units = 'months since 2022-01-01'
time_var.calendar = 'gregorian'
time_var.long_name = 'time'
lat_var.units = 'degrees_north'
lat_var.long_name = 'latitude'
lon_var.units = 'degrees_east'
lon_var.long_name = 'longitude'
temp_var.units = 'Kelvin'
temp_var.long_name = 'air_temperature'

for month in range(12):
    seasonal_factor = -12 * np.cos(2 * np.pi * month / 12)
    for i, lat in enumerate(lat_var[:]):
        for j, lon in enumerate(lon_var[:]):
            base_temp = 286.15 - (lat - 40.0) * 0.5
            temp = base_temp + seasonal_factor
            maritime_effect = 2.0 * np.cos(np.radians(lon * 2))
            temp += maritime_effect
            np.random.seed(2022 * 100 + month * 1000 + i * 10 + j)
            noise = np.random.normal(0, 1.5)
            temp += noise
            temp_var[month, i, j] = temp

nc_file.title = 'Monthly Temperature Data 2022'
nc_file.institution = 'Meteorological Agency'
nc_file.source = 'Climate Model'
nc_file.history = 'Created for weather station reporting'
nc_file.close()

print("temp_2022.nc created successfully")

# Create temp_2023.nc
nc_file = Dataset('/data/climate/temp_2023.nc', 'w', format='NETCDF4')

time_dim = nc_file.createDimension('time', 12)
lat_dim = nc_file.createDimension('latitude', 21)
lon_dim = nc_file.createDimension('longitude', 41)

time_var = nc_file.createVariable('time', 'f4', ('time',))
lat_var = nc_file.createVariable('latitude', 'f4', ('latitude',))
lon_var = nc_file.createVariable('longitude', 'f4', ('longitude',))
temp_var = nc_file.createVariable('temperature', 'f4', ('time', 'latitude', 'longitude'))

time_var[:] = np.arange(0, 12)
lat_var[:] = np.arange(40.0, 50.5, 0.5)
lon_var[:] = np.arange(-10.0, 10.5, 0.5)

time_var.units = 'months since 2023-01-01'
time_var.calendar = 'gregorian'
time_var.long_name = 'time'
lat_var.units = 'degrees_north'
lat_var.long_name = 'latitude'
lon_var.units = 'degrees_east'
lon_var.long_name = 'longitude'
temp_var.units = 'Kelvin'
temp_var.long_name = 'air_temperature'

for month in range(12):
    seasonal_factor = -12 * np.cos(2 * np.pi * month / 12)
    for i, lat in enumerate(lat_var[:]):
        for j, lon in enumerate(lon_var[:]):
            base_temp = 286.15 - (lat - 40.0) * 0.5
            temp = base_temp + seasonal_factor
            maritime_effect = 2.0 * np.cos(np.radians(lon * 2))
            temp += maritime_effect
            np.random.seed(2023 * 100 + month * 1000 + i * 10 + j)
            noise = np.random.normal(0, 1.5)
            temp += noise
            temp_var[month, i, j] = temp

nc_file.title = 'Monthly Temperature Data 2023'
nc_file.institution = 'Meteorological Agency'
nc_file.source = 'Climate Model'
nc_file.history = 'Created for weather station reporting'
nc_file.close()

print("temp_2023.nc created successfully")

# Now process the data to extract station time series
from netCDF4 import Dataset
import numpy as np
import os

# Station coordinates
stations = {
    'A': (45.0, 0.0),
    'B': (42.5, 5.0),
    'C': (48.0, -3.5),
    'D': (41.0, 8.5),
    'E': (49.5, -7.0)
}

# Function to find nearest grid point
def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# Dictionary to store all temperature data for each station
station_temps = {station: [] for station in stations.keys()}

# Process each year
for year in [2021, 2022, 2023]:
    nc_file = Dataset(f'/data/climate/temp_{year}.nc', 'r')
    
    lat_values = nc_file.variables['latitude'][:]
    lon_values = nc_file.variables['longitude'][:]
    temp_data = nc_file.variables['temperature'][:]
    
    # Extract data for each station
    for station_id, (lat, lon) in stations.items():
        lat_idx = find_nearest_index(lat_values, lat)
        lon_idx = find_nearest_index(lon_values, lon)
        
        # Extract time series for this year (12 months)
        temps_kelvin = temp_data[:, lat_idx, lon_idx]
        
        # Convert to Celsius
        temps_celsius = temps_kelvin - 273.15
        
        # Append to station's time series
        station_temps[station_id].extend(temps_celsius)
    
    nc_file.close()

# Calculate 3-year averages
station_averages = {}
for station_id, temps in station_temps.items():
    avg_temp = np.mean(temps)
    station_averages[station_id] = avg_temp

# Find the station with highest average temperature
max_station = max(station_averages, key=station_averages.get)

# Create output directory
os.makedirs('/output', exist_ok=True)

# Write results to file
with open('/output/station_summary.txt', 'w') as f:
    f.write('STATION_ID,AVG_TEMP_C,HIGHEST\n')
    
    for station_id in sorted(stations.keys()):
        avg_temp = station_averages[station_id]
        is_highest = 'YES' if station_id == max_station else 'NO'
        f.write(f'{station_id},{avg_temp:.2f},{is_highest}\n')

print("Station summary created successfully")