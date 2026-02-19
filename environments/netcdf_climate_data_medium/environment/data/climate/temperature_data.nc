import netCDF4 as nc
import numpy as np
import json
import os

# Create the NetCDF file
os.makedirs('/data/climate', exist_ok=True)
nc_file = nc.Dataset('/data/climate/temperature_data.nc', 'w', format='NETCDF4')

# Create dimensions
station_dim = nc_file.createDimension('station', 15)
time_dim = nc_file.createDimension('time', 120)

# Create variables
temperature_var = nc_file.createVariable('temperature', 'f4', ('station', 'time'))
station_name_var = nc_file.createVariable('station_name', str, ('station',))
latitude_var = nc_file.createVariable('latitude', 'f4', ('station',))
longitude_var = nc_file.createVariable('longitude', 'f4', ('station',))

# Generate station names
station_names = [
    'Phoenix_AZ', 'Miami_FL', 'Las_Vegas_NV', 'Houston_TX', 'Atlanta_GA',
    'Denver_CO', 'Seattle_WA', 'Boston_MA', 'Chicago_IL', 'Anchorage_AK',
    'Portland_OR', 'Minneapolis_MN', 'Detroit_MI', 'Dallas_TX', 'San_Diego_CA'
]

# Generate latitudes and longitudes (realistic values)
latitudes = np.array([33.4, 25.8, 36.2, 29.8, 33.7, 39.7, 47.6, 42.4, 41.9, 61.2, 45.5, 45.0, 42.3, 32.8, 32.7])
longitudes = np.array([-112.1, -80.2, -115.1, -95.4, -84.4, -104.9, -122.3, -71.1, -87.6, -149.9, -122.7, -93.3, -83.0, -96.8, -117.2])

# Generate temperature data
np.random.seed(42)
temperature_data = np.zeros((15, 120), dtype=np.float32)

# For each station, generate realistic temperature patterns
for i in range(15):
    # Base temperature varies by latitude
    if latitudes[i] > 60:  # Cold climate (Anchorage)
        base_temp = 5.0
        amplitude = 15.0
        min_offset = -20.0
    elif latitudes[i] > 40:  # Temperate climate
        base_temp = 15.0
        amplitude = 15.0
        min_offset = -10.0
    elif latitudes[i] > 30:  # Warm temperate
        base_temp = 20.0
        amplitude = 12.0
        min_offset = 0.0
    else:  # Hot climate
        base_temp = 25.0
        amplitude = 10.0
        min_offset = 5.0
    
    # Create seasonal pattern (120 time points representing ~10 years of monthly data)
    time_points = np.linspace(0, 10 * 2 * np.pi, 120)
    seasonal_pattern = base_temp + amplitude * np.sin(time_points)
    
    # Add random variation
    noise = np.random.normal(0, 3, 120)
    temperature_data[i, :] = seasonal_pattern + noise + min_offset

# Ensure at least 8 stations have readings above 35°C
stations_above_35 = []
for i in range(15):
    if np.max(temperature_data[i, :]) > 35:
        stations_above_35.append(i)

# If not enough stations above 35°C, add hot temperatures to specific stations
while len(stations_above_35) < 8:
    for i in [0, 1, 2, 3, 13, 14, 4, 6]:  # Hot climate stations
        if i not in stations_above_35:
            # Add some hot summer temperatures
            hot_indices = np.random.choice(range(120), size=5, replace=False)
            temperature_data[i, hot_indices] = np.random.uniform(36, 44, 5)
            if np.max(temperature_data[i, :]) > 35:
                stations_above_35.append(i)
            if len(stations_above_35) >= 8:
                break

# Ensure we have appropriate min and max values
# Set minimum around -12 to -15
cold_station_idx = 9  # Anchorage
cold_time_idx = np.argmin(temperature_data[cold_station_idx, :])
temperature_data[cold_station_idx, cold_time_idx] = -13.5

# Set maximum around 42-45
hot_station_idx = 0  # Phoenix
hot_time_idx = np.argmax(temperature_data[hot_station_idx, :])
temperature_data[hot_station_idx, hot_time_idx] = 43.2

# Write data to variables
temperature_var[:] = temperature_data
station_name_var[:] = station_names
latitude_var[:] = latitudes
longitude_var[:] = longitudes

# Close the NetCDF file
nc_file.close()

# Now read the NetCDF file and create the summary
nc_file = nc.Dataset('/data/climate/temperature_data.nc', 'r')

# Extract information
station_count = len(nc_file.dimensions['station'])
observations_per_station = len(nc_file.dimensions['time'])

# Read temperature data
temp_data = nc_file.variables['temperature'][:]

# Calculate min and max
min_temperature = float(np.min(temp_data))
max_temperature = float(np.max(temp_data))

# Count stations with at least one reading above 35°C
stations_above_35c = 0
for i in range(station_count):
    if np.max(temp_data[i, :]) > 35:
        stations_above_35c += 1

nc_file.close()

# Create summary dictionary
summary = {
    "station_count": station_count,
    "observations_per_station": observations_per_station,
    "min_temperature": round(min_temperature, 1),
    "max_temperature": round(max_temperature, 1),
    "stations_above_35c": stations_above_35c
}

# Save to JSON file
os.makedirs('/output', exist_ok=True)
with open('/output/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("Summary created successfully!")
print(json.dumps(summary, indent=2))