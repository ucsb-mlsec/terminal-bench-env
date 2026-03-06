class GeoPoint:
    def __init__(self, latitude, longitude, metadata=None):
        self.latitude = latitude
        self.longitude = longitude
        self.metadata = metadata if metadata is not None else {}
    
    def __eq__(self, other):
        if not isinstance(other, GeoPoint):
            return False
        return (self.latitude == other.latitude and 
                self.longitude == other.longitude and 
                self.metadata == other.metadata)
    
    def __repr__(self):
        return f'GeoPoint(lat={self.latitude}, lon={self.longitude}, metadata={self.metadata})'