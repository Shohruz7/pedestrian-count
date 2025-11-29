"""
Location API routes
"""
from flask import Blueprint, request, jsonify
from sqlalchemy import and_, or_, func
from app import db
from app.models import Location, AggregatedCount
from app.services.map_service import create_geojson_featurecollection

bp = Blueprint('locations', __name__)


@bp.route('', methods=['GET'])
def get_locations():
    """Get all locations with optional filters"""
    try:
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


@bp.route('/<int:location_id>', methods=['GET'])
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

