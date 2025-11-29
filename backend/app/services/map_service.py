"""
Map service utilities
"""
from app.models import Location


def create_geojson_featurecollection(locations):
    """Create a GeoJSON FeatureCollection from locations"""
    features = []
    for loc in locations:
        geojson = loc.to_geojson()
        if geojson:
            features.append(geojson)
    
    return {
        'type': 'FeatureCollection',
        'features': features
    }

