"""
Statistics API routes
"""
from flask import Blueprint, request, jsonify
from sqlalchemy import func, and_, or_
from app import db, limiter
from app.models import Location, AggregatedCount

bp = Blueprint('statistics', __name__)


@bp.route('/summary', methods=['GET'])
@limiter.limit("60 per minute")
def get_summary():
    """Get overall summary statistics"""
    try:
        # Get filter parameters
        boroughs = request.args.getlist('borough[]')
        categories = request.args.getlist('category[]')
        min_count = request.args.get('min_count', type=float)
        max_count = request.args.get('max_count', type=float)
        
        # Build query
        query = db.session.query(
            func.count(Location.id).label('total_locations'),
            func.avg(AggregatedCount.avg_recent_count).label('mean_count'),
            func.min(AggregatedCount.avg_recent_count).label('min_count'),
            func.max(AggregatedCount.avg_recent_count).label('max_count'),
            func.stddev(AggregatedCount.avg_recent_count).label('std_dev')
        ).join(
            AggregatedCount, Location.id == AggregatedCount.location_id
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
        
        result = query.first()
        
        # Calculate median
        median_query = db.session.query(
            func.percentile_cont(0.5).within_group(
                AggregatedCount.avg_recent_count
            ).label('median')
        ).join(
            Location, AggregatedCount.location_id == Location.id
        )
        
        if boroughs:
            median_query = median_query.filter(Location.borough.in_(boroughs))
        if categories:
            median_query = median_query.filter(Location.category.in_(categories))
        if min_count is not None:
            median_query = median_query.filter(AggregatedCount.avg_recent_count >= min_count)
        if max_count is not None:
            median_query = median_query.filter(AggregatedCount.avg_recent_count <= max_count)
        
        median = median_query.scalar()
        
        return jsonify({
            'total_locations': result.total_locations or 0,
            'mean_count': float(result.mean_count) if result.mean_count else 0,
            'median_count': float(median) if median else 0,
            'min_count': float(result.min_count) if result.min_count else 0,
            'max_count': float(result.max_count) if result.max_count else 0,
            'std_dev': float(result.std_dev) if result.std_dev else 0
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/by-borough', methods=['GET'])
@limiter.limit("60 per minute")
def get_by_borough():
    """Get statistics grouped by borough"""
    try:
        query = db.session.query(
            Location.borough,
            func.count(Location.id).label('location_count'),
            func.avg(AggregatedCount.avg_recent_count).label('avg_count'),
            func.min(AggregatedCount.avg_recent_count).label('min_count'),
            func.max(AggregatedCount.avg_recent_count).label('max_count'),
            func.stddev(AggregatedCount.avg_recent_count).label('std_dev')
        ).join(
            AggregatedCount, Location.id == AggregatedCount.location_id
        ).group_by(Location.borough)
        
        results = query.all()
        
        stats = []
        for row in results:
            # Calculate median for this borough
            median_query = db.session.query(
                func.percentile_cont(0.5).within_group(
                    AggregatedCount.avg_recent_count
                ).label('median')
            ).join(
                Location, AggregatedCount.location_id == Location.id
            ).filter(Location.borough == row.borough)
            
            median = median_query.scalar()
            
            stats.append({
                'borough': row.borough,
                'location_count': row.location_count,
                'avg_count': float(row.avg_count) if row.avg_count else 0,
                'median_count': float(median) if median else 0,
                'min_count': float(row.min_count) if row.min_count else 0,
                'max_count': float(row.max_count) if row.max_count else 0,
                'std_dev': float(row.std_dev) if row.std_dev else 0
            })
        
        return jsonify({'statistics': stats}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/by-category', methods=['GET'])
@limiter.limit("60 per minute")
def get_by_category():
    """Get statistics grouped by category"""
    try:
        query = db.session.query(
            Location.category,
            func.count(Location.id).label('location_count'),
            func.avg(AggregatedCount.avg_recent_count).label('avg_count'),
            func.min(AggregatedCount.avg_recent_count).label('min_count'),
            func.max(AggregatedCount.avg_recent_count).label('max_count'),
            func.stddev(AggregatedCount.avg_recent_count).label('std_dev')
        ).join(
            AggregatedCount, Location.id == AggregatedCount.location_id
        ).group_by(Location.category)
        
        results = query.all()
        
        stats = []
        for row in results:
            # Calculate median for this category
            median_query = db.session.query(
                func.percentile_cont(0.5).within_group(
                    AggregatedCount.avg_recent_count
                ).label('median')
            ).join(
                Location, AggregatedCount.location_id == Location.id
            ).filter(Location.category == row.category)
            
            median = median_query.scalar()
            
            stats.append({
                'category': row.category,
                'location_count': row.location_count,
                'avg_count': float(row.avg_count) if row.avg_count else 0,
                'median_count': float(median) if median else 0,
                'min_count': float(row.min_count) if row.min_count else 0,
                'max_count': float(row.max_count) if row.max_count else 0,
                'std_dev': float(row.std_dev) if row.std_dev else 0
            })
        
        return jsonify({'statistics': stats}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/top-sites', methods=['GET'])
@limiter.limit("60 per minute")
def get_top_sites():
    """Get top N sites by pedestrian count"""
    try:
        limit = request.args.get('limit', 10, type=int)
        if limit > 100:
            limit = 100  # Cap at 100
        
        query = db.session.query(
            Location,
            AggregatedCount.avg_recent_count
        ).join(
            AggregatedCount, Location.id == AggregatedCount.location_id
        ).order_by(
            AggregatedCount.avg_recent_count.desc()
        ).limit(limit)
        
        results = query.all()
        
        sites = []
        for location, avg_count in results:
            sites.append({
                'location': location.to_dict(),
                'avg_recent_count': float(avg_count) if avg_count else 0
            })
        
        return jsonify({
            'sites': sites,
            'count': len(sites)
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

