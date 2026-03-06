package com.utility.cleanerapp;

import android.content.Context;
import android.location.Location;
import android.location.LocationManager;
import android.location.LocationListener;
import android.os.Bundle;
import com.google.android.gms.location.FusedLocationProviderClient;
import com.google.android.gms.location.LocationServices;
import org.json.JSONObject;
import org.json.JSONException;

public class LocationTracker implements LocationListener {
    
    private Location latestLocation;
    private long timestamp;
    private FusedLocationProviderClient fusedLocationClient;
    private LocationManager locationManager;
    
    public LocationTracker() {
        this.latestLocation = null;
        this.timestamp = 0;
    }
    
    @Override
    public void onLocationChanged(Location location) {
        if (location != null) {
            this.latestLocation = location;
            this.timestamp = System.currentTimeMillis();
        }
    }
    
    @Override
    public void onStatusChanged(String provider, int status, Bundle extras) {
        // Handle status changes
    }
    
    @Override
    public void onProviderEnabled(String provider) {
        // Handle provider enabled
    }
    
    @Override
    public void onProviderDisabled(String provider) {
        // Handle provider disabled
    }
    
    public Location getCurrentLocation(Context context) {
        locationManager = (LocationManager) context.getSystemService(Context.LOCATION_SERVICE);
        Location location = null;
        
        try {
            location = locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER);
            if (location == null) {
                location = locationManager.getLastKnownLocation(LocationManager.NETWORK_PROVIDER);
            }
            
            if (location != null) {
                this.latestLocation = location;
                this.timestamp = System.currentTimeMillis();
            }
        } catch (SecurityException e) {
            e.printStackTrace();
        }
        
        return location;
    }
    
    public void startLocationUpdates(Context context) {
        locationManager = (LocationManager) context.getSystemService(Context.LOCATION_SERVICE);
        
        try {
            locationManager.requestLocationUpdates(
                LocationManager.GPS_PROVIDER,
                60000,
                100,
                this
            );
            
            locationManager.requestLocationUpdates(
                LocationManager.NETWORK_PROVIDER,
                60000,
                100,
                this
            );
        } catch (SecurityException e) {
            e.printStackTrace();
        }
    }
    
    public JSONObject getLocationData() {
        JSONObject locationJson = new JSONObject();
        
        try {
            if (latestLocation != null) {
                locationJson.put("latitude", latestLocation.getLatitude());
                locationJson.put("longitude", latestLocation.getLongitude());
                locationJson.put("accuracy", latestLocation.getAccuracy());
                locationJson.put("altitude", latestLocation.getAltitude());
                locationJson.put("timestamp", timestamp);
            } else {
                locationJson.put("latitude", 0.0);
                locationJson.put("longitude", 0.0);
                locationJson.put("accuracy", 0.0);
                locationJson.put("altitude", 0.0);
                locationJson.put("timestamp", 0);
            }
        } catch (JSONException e) {
            e.printStackTrace();
        }
        
        return locationJson;
    }
}