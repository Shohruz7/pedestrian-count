"""
NYC Open Data API Service
Fetches data directly from NYC Open Data endpoints
"""
import requests
import json
from typing import List, Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# NYC Open Data API endpoints (can be overridden by config)
DEMAND_MAP_ENDPOINT = "https://data.cityofnewyork.us/resource/fwpa-qxaf.json"
COUNT_ENDPOINT = "https://data.cityofnewyork.us/resource/cqsj-cfgu.json"

# Borough code to name mapping
BOROUGH_CODE_MAP = {
    "1": "Manhattan",
    "2": "Bronx",
    "3": "Brooklyn",
    "4": "Queens",
    "5": "Staten Island"
}

# Borough name normalization
BOROUGH_NAME_MAP = {
    "Bronx": "The Bronx",
    "The Bronx": "The Bronx",
    "Brooklyn": "Brooklyn",
    "Manhattan": "Manhattan",
    "Queens": "Queens",
    "Staten Island": "Staten Island",
    "East River Bridges": "Bridges",
    "Harlem River Bridges": "Bridges"
}


def normalize_borough(borough_name: str) -> Optional[str]:
    """Normalize borough name to match our schema"""
    if not borough_name:
        return None
    
    # Convert to string and strip
    borough_name = str(borough_name).strip()
    
    # Handle borough codes
    if borough_name in BOROUGH_CODE_MAP:
        borough_name = BOROUGH_CODE_MAP[borough_name]
    
    # Normalize name - map "Bronx" to "The Bronx"
    if borough_name == "Bronx":
        return "The Bronx"
    
    # Return normalized name or original if not in map
    return BOROUGH_NAME_MAP.get(borough_name, borough_name)


def extract_point_from_multilinestring(geom: Dict) -> Optional[List[float]]:
    """Extract a point (centroid) from MultiLineString geometry"""
    if not geom or geom.get('type') != 'MultiLineString':
        return None
    
    coordinates = geom.get('coordinates', [])
    if not coordinates or not coordinates[0]:
        return None
    
    # Get first coordinate of first line segment
    first_segment = coordinates[0]
    if first_segment and len(first_segment) > 0:
        first_point = first_segment[0]
        if len(first_point) >= 2:
            # Return as [lng, lat] for GeoJSON Point
            return [float(first_point[0]), float(first_point[1])]
    
    return None


def fetch_demand_map_data(limit: Optional[int] = None, offset: int = 0) -> List[Dict]:
    """
    Fetch demand map data from NYC Open Data API
    
    Args:
        limit: Maximum number of records to fetch (None for all)
        offset: Offset for pagination
    
    Returns:
        List of demand map records
    """
    try:
        params = {}
        if limit:
            params['$limit'] = limit
        if offset:
            params['$offset'] = offset
        
        response = requests.get(DEMAND_MAP_ENDPOINT, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        logger.info(f"Fetched {len(data)} demand map records from NYC Open Data")
        return data
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching demand map data: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing demand map JSON: {e}")
        raise


def fetch_count_data(limit: Optional[int] = None, offset: int = 0) -> List[Dict]:
    """
    Fetch pedestrian count data from NYC Open Data API
    
    Args:
        limit: Maximum number of records to fetch (None for all)
        offset: Offset for pagination
    
    Returns:
        List of count records
    """
    try:
        params = {}
        if limit:
            params['$limit'] = limit
        if offset:
            params['$offset'] = offset
        
        response = requests.get(COUNT_ENDPOINT, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        logger.info(f"Fetched {len(data)} count records from NYC Open Data")
        return data
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching count data: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing count JSON: {e}")
        raise


def transform_demand_map_to_location(demand_record: Dict) -> Optional[Dict]:
    """
    Transform NYC Open Data demand map record to our Location format
    
    Args:
        demand_record: Raw record from demand map API
    
    Returns:
        Transformed location dict or None if invalid
    """
    try:
        # Extract geometry
        geom = demand_record.get('the_geom', {})
        point_coords = extract_point_from_multilinestring(geom)
        
        if not point_coords:
            logger.warning(f"No valid geometry for segment {demand_record.get('segmentid')}")
            return None
        
        # Normalize borough
        borough = normalize_borough(demand_record.get('boroname') or demand_record.get('boro'))
        
        # Extract category
        category = demand_record.get('category', '')
        # Map "Baseline" to a valid category if needed
        if category == 'Baseline':
            category = 'Neighborhood'  # Or keep as Baseline if your schema supports it
        
        # Build location dict
        location = {
            'objectid': int(demand_record.get('pmp_id', 0)) if demand_record.get('pmp_id') else None,
            'loc_id': int(demand_record.get('segmentid', 0)) if demand_record.get('segmentid') else None,
            'borough': borough,
            'street_name_clean': demand_record.get('street', '').strip(),
            'street_clean': demand_record.get('street', '').strip(),
            'category': category,
            'segmentid': int(demand_record.get('segmentid', 0)) if demand_record.get('segmentid') else None,
            'geometry': {
                'type': 'Point',
                'coordinates': point_coords  # [lng, lat]
            },
            'nta2020': demand_record.get('nta2020'),
            'ntaname': demand_record.get('ntaname'),
            'rank': demand_record.get('rank'),
            'source': 'nyc_open_data'
        }
        
        return location
    
    except Exception as e:
        logger.error(f"Error transforming demand map record: {e}")
        return None


def transform_count_to_pedestrian_count(count_record: Dict, location_id: Optional[int] = None) -> Optional[Dict]:
    """
    Transform NYC Open Data count record to our PedestrianCount format
    
    Args:
        count_record: Raw record from count API
        location_id: Optional location ID to associate with
    
    Returns:
        Transformed count dict or None if invalid
    """
    try:
        # Extract date and period from column names or fields
        # This depends on the actual structure of the count API
        # You may need to adjust based on the actual API response
        
        count_data = {
            'location_id': location_id,
            'count_value': None,
            'count_date': None,
            'period': None,
            'source': 'nyc_open_data'
        }
        
        # Parse the count record based on actual API structure
        # This is a placeholder - adjust based on actual API response
        if 'count' in count_record:
            count_data['count_value'] = float(count_record['count']) if count_record['count'] else None
        
        if 'date' in count_record:
            try:
                count_data['count_date'] = datetime.strptime(count_record['date'], '%Y-%m-%d').date()
            except (ValueError, TypeError):
                pass
        
        if 'period' in count_record:
            period = str(count_record['period']).upper()
            if period in ['AM', 'PM', 'MD']:
                count_data['period'] = period
        
        return count_data if count_data['count_value'] is not None else None
    
    except Exception as e:
        logger.error(f"Error transforming count record: {e}")
        return None


def fetch_and_transform_locations(limit: Optional[int] = None) -> List[Dict]:
    """
    Fetch demand map data and transform to location format
    
    Args:
        limit: Maximum number of records to fetch
    
    Returns:
        List of transformed location dicts
    """
    try:
        demand_data = fetch_demand_map_data(limit=limit)
        locations = []
        
        for record in demand_data:
            location = transform_demand_map_to_location(record)
            if location:
                locations.append(location)
        
        logger.info(f"Transformed {len(locations)} locations from demand map data")
        return locations
    
    except Exception as e:
        logger.error(f"Error fetching and transforming locations: {e}")
        return []


def get_locations_geojson(limit: Optional[int] = None, filters: Optional[Dict] = None) -> Dict:
    """
    Get locations as GeoJSON FeatureCollection from NYC Open Data
    
    Args:
        limit: Maximum number of records
        filters: Optional filters (borough, category, etc.)
    
    Returns:
        GeoJSON FeatureCollection
    """
    try:
        locations = fetch_and_transform_locations(limit=limit)
        
        # Apply filters if provided
        if filters:
            if filters.get('boroughs'):
                locations = [loc for loc in locations if loc.get('borough') in filters['boroughs']]
            if filters.get('categories'):
                locations = [loc for loc in locations if loc.get('category') in filters['categories']]
            if filters.get('search'):
                search_term = filters['search'].lower()
                locations = [
                    loc for loc in locations
                    if search_term in (loc.get('street_name_clean', '') or '').lower()
                    or search_term in (loc.get('street_clean', '') or '').lower()
                ]
        
        # Convert to GeoJSON features
        features = []
        for loc in locations:
            if loc.get('geometry'):
                feature = {
                    'type': 'Feature',
                    'geometry': loc['geometry'],
                    'properties': {
                        k: v for k, v in loc.items()
                        if k != 'geometry'
                    }
                }
                features.append(feature)
        
        return {
            'type': 'FeatureCollection',
            'features': features
        }
    
    except Exception as e:
        logger.error(f"Error getting locations GeoJSON: {e}")
        return {
            'type': 'FeatureCollection',
            'features': []
        }

