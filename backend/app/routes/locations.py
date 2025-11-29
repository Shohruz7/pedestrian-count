"""
Location API routes
"""
from flask import Blueprint, request, jsonify, current_app
from sqlalchemy import and_, or_, func
from app import db, limiter
from app.models import Location, AggregatedCount
from app.services.map_service import create_geojson_featurecollection
from app.services.nyc_api_service import get_locations_geojson, fetch_and_transform_locations

bp = Blueprint('locations', __name__)


@bp.route('', methods=['GET'])
@limiter.limit("30 per minute")
def get_locations():
    """Get all locations with optional filters"""
    try:
        # Check if we should use NYC API
        use_nyc_api = request.args.get('source', '').lower() == 'nyc' or current_app.config.get('USE_NYC_API', False)
        
        if use_nyc_api:
            return get_locations_from_nyc_api()
        
        # Get query parameters
        boroughs = request.args.getlist('borough[]')
        categories = request.args.getlist('category[]')
        min_count = request.args.get('min_count', type=float)
        max_count = request.args.get('max_count', type=float)
        search = request.args.get('search', '').strip()
        bounds = request.args.get('bounds')  # Format: "min_lat,min_lon,max_lat,max_lon"
        format_type = request.args.get('format', 'json')  # 'json' or 'geojson'
        
        # Build query
        query = db.session.query(Location).join(
            AggregatedCount, Location.id == AggregatedCount.location_id, isouter=True
        )
        
        # Apply filters
        if boroughs:
            query = query.filter(Location.borough.in_(boroughs))
        
        if categories:
            query = query.filter(Location.category.in_(categories))
        
        if min_count is not None:
            query = query.filter(AggregatedCount.avg_recent_count >= min_count)
        
        if max_count is not None:
            query = query.filter(AggregatedCount.avg_recent_count <= max_count)
        
        if search:
            search_pattern = f'%{search}%'
            query = query.filter(
                or_(
                    Location.street_name_clean.ilike(search_pattern),
                    Location.street_clean.ilike(search_pattern),
                    Location.loc_id.cast(db.String).ilike(search_pattern)
                )
            )
        
        # Apply bounding box filter if provided
        if bounds:
            try:
                coords = [float(c) for c in bounds.split(',')]
                if len(coords) == 4:
                    min_lat, min_lon, max_lat, max_lon = coords
                    query = query.filter(
                        func.ST_Within(
                            Location.geometry,
                            func.ST_MakeEnvelope(min_lon, min_lat, max_lon, max_lat, 4326)
                        )
                    )
            except (ValueError, IndexError):
                pass  # Invalid bounds format, ignore
        
        # Execute query
        locations = query.all()
        
        # Format response
        if format_type == 'geojson':
            features = []
            for loc in locations:
                geojson = loc.to_geojson()
                if geojson:
                    features.append(geojson)
            
            return jsonify({
                'type': 'FeatureCollection',
                'features': features
            }), 200
        else:
            # Return JSON format
            return jsonify({
                'locations': [loc.to_dict() for loc in locations],
                'count': len(locations)
            }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def get_locations_from_nyc_api():
    """Get locations from NYC Open Data API"""
    try:
        # Get query parameters
        boroughs = request.args.getlist('borough[]')
        categories = request.args.getlist('category[]')
        search = request.args.get('search', '').strip()
        format_type = request.args.get('format', 'geojson')
        limit = request.args.get('limit', type=int)
        
        # Build filters dict
        filters = {}
        if boroughs:
            filters['boroughs'] = boroughs
        if categories:
            filters['categories'] = categories
        if search:
            filters['search'] = search
        
        # Fetch from NYC API
        geojson_data = get_locations_geojson(limit=limit, filters=filters if filters else None)
        
        if format_type == 'geojson':
            return jsonify(geojson_data), 200
        else:
            # Convert GeoJSON to JSON format
            locations = []
            for feature in geojson_data.get('features', []):
                props = feature.get('properties', {})
                locations.append({
                    'id': props.get('objectid'),
                    'objectid': props.get('objectid'),
                    'loc_id': props.get('loc_id'),
                    'borough': props.get('borough'),
                    'street_name_clean': props.get('street_name_clean'),
                    'street_clean': props.get('street_clean'),
                    'category': props.get('category'),
                    'segmentid': props.get('segmentid'),
                })
            
            return jsonify({
                'locations': locations,
                'count': len(locations),
                'source': 'nyc_open_data'
            }), 200
    
    except Exception as e:
        return jsonify({'error': f'Error fetching from NYC API: {str(e)}'}), 500


@bp.route('/<int:location_id>', methods=['GET'])
@limiter.limit("30 per minute")
def get_location(location_id):
    """Get a single location by ID"""
    try:
        location = db.session.query(Location).filter_by(id=location_id).first()
        
        if not location:
            return jsonify({'error': 'Location not found'}), 404
        
        format_type = request.args.get('format', 'json')
        
        if format_type == 'geojson':
            geojson = location.to_geojson()
            if geojson:
                return jsonify(geojson), 200
            else:
                return jsonify({'error': 'Location has no geometry'}), 400
        else:
            return jsonify(location.to_dict()), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/bounds', methods=['GET'])
@limiter.limit("30 per minute")
def get_locations_in_bounds():
    """Get locations within map bounds (for performance optimization)"""
    try:
        bounds = request.args.get('bounds')
        if not bounds:
            return jsonify({'error': 'bounds parameter required'}), 400
        
        coords = [float(c) for c in bounds.split(',')]
        if len(coords) != 4:
            return jsonify({'error': 'Invalid bounds format. Use: min_lat,min_lon,max_lat,max_lon'}), 400
        
        min_lat, min_lon, max_lat, max_lon = coords
        
        # Get additional filters
        boroughs = request.args.getlist('borough[]')
        categories = request.args.getlist('category[]')
        min_count = request.args.get('min_count', type=float)
        max_count = request.args.get('max_count', type=float)
        
        query = db.session.query(Location).join(
            AggregatedCount, Location.id == AggregatedCount.location_id, isouter=True
        ).filter(
            func.ST_Within(
                Location.geometry,
                func.ST_MakeEnvelope(min_lon, min_lat, max_lon, max_lat, 4326)
            )
        )
        
        if boroughs:
            query = query.filter(Location.borough.in_(boroughs))
        if categories:
            query = query.filter(Location.category.in_(categories))
        if min_count is not None:
            query = query.filter(AggregatedCount.avg_recent_count >= min_count)
        if max_count is not None:
            query = query.filter(AggregatedCount.avg_recent_count <= max_count)
        
        locations = query.all()
        
        features = []
        for loc in locations:
            geojson = loc.to_geojson()
            if geojson:
                features.append(geojson)
        
        return jsonify({
            'type': 'FeatureCollection',
            'features': features
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/nyc', methods=['GET'])
@limiter.limit("20 per minute")
def get_locations_nyc():
    """Get locations directly from NYC Open Data API"""
    return get_locations_from_nyc_api()

