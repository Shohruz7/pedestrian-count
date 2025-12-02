"""
Export API routes
"""
from flask import Blueprint, request, jsonify, Response
from sqlalchemy import and_, or_, func
from app import db, limiter
from app.models import Location, AggregatedCount, PedestrianCount
import csv
import io
import json

bp = Blueprint('export', __name__)


@bp.route('/csv', methods=['GET'])
@limiter.limit("20 per minute")
def export_csv():
    """Export filtered data as CSV"""
    try:
        # Get filter parameters (same as locations endpoint)
        boroughs = request.args.getlist('borough[]')
        categories = request.args.getlist('category[]')
        min_count = request.args.get('min_count', type=float)
        max_count = request.args.get('max_count', type=float)
        search = request.args.get('search', '').strip()
        
        # Build query
        query = db.session.query(
            Location,
            AggregatedCount.avg_recent_count,
            AggregatedCount.min_count,
            AggregatedCount.max_count,
            AggregatedCount.std_dev
        ).join(
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
        
        results = query.all()
        
        # Create CSV
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'ID', 'ObjectID', 'Loc ID', 'Borough', 'Street Name', 'Street',
            'Category', 'Segment ID', 'Avg Count', 'Min Count', 'Max Count', 'Std Dev'
        ])
        
        # Write data
        for location, avg_count, min_count, max_count, std_dev in results:
            writer.writerow([
                location.id,
                location.objectid,
                location.loc_id or '',
                location.borough or '',
                location.street_name_clean or '',
                location.street_clean or '',
                location.category or '',
                location.segmentid or '',
                float(avg_count) if avg_count else '',
                float(min_count) if min_count else '',
                float(max_count) if max_count else '',
                float(std_dev) if std_dev else ''
            ])
        
        # Create response
        response = Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=pedestrian_data.csv'}
        )
        
        return response
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/geojson', methods=['GET'])
@limiter.limit("20 per minute")
def export_geojson():
    """Export filtered data as GeoJSON"""
    try:
        # Get filter parameters
        boroughs = request.args.getlist('borough[]')
        categories = request.args.getlist('category[]')
        min_count = request.args.get('min_count', type=float)
        max_count = request.args.get('max_count', type=float)
        search = request.args.get('search', '').strip()
        
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
        
        locations = query.all()
        
        # Create GeoJSON
        features = []
        for loc in locations:
            geojson = loc.to_geojson()
            if geojson:
                features.append(geojson)
        
        geojson_data = {
            'type': 'FeatureCollection',
            'features': features
        }
        
        response = Response(
            json.dumps(geojson_data),
            mimetype='application/json',
            headers={'Content-Disposition': 'attachment; filename=pedestrian_data.geojson'}
        )
        
        return response
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

