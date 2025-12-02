"""
Comparison API routes
"""
from flask import Blueprint, request, jsonify
from sqlalchemy import func, and_, or_
from app import db, limiter
from app.models import Location, AggregatedCount

bp = Blueprint('comparison', __name__)


@bp.route('', methods=['POST'])
@limiter.limit("30 per minute")
def compare_groups():
    """Compare two groups (by borough or category)"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Request body required'}), 400
        
        group1 = data.get('group1', {})
        group2 = data.get('group2', {})
        
        if not group1 or not group2:
            return jsonify({'error': 'Both group1 and group2 required'}), 400
        
        # Get statistics for group 1
        stats1 = _get_group_stats(group1)
        if 'error' in stats1:
            return jsonify(stats1), 400
        
        # Get statistics for group 2
        stats2 = _get_group_stats(group2)
        if 'error' in stats2:
            return jsonify(stats2), 400
        
        # Calculate differences
        count_diff = stats1['count'] - stats2['count']
        count_diff_pct = ((stats1['count'] - stats2['count']) / stats2['count'] * 100) if stats2['count'] > 0 else 0
        
        mean_diff = stats1['mean'] - stats2['mean']
        mean_diff_pct = ((stats1['mean'] - stats2['mean']) / stats2['mean'] * 100) if stats2['mean'] > 0 else 0
        
        median_diff = stats1['median'] - stats2['median']
        median_diff_pct = ((stats1['median'] - stats2['median']) / stats2['median'] * 100) if stats2['median'] > 0 else 0
        
        max_diff = stats1['max'] - stats2['max']
        max_diff_pct = ((stats1['max'] - stats2['max']) / stats2['max'] * 100) if stats2['max'] > 0 else 0
        
        return jsonify({
            'group1': {
                'type': group1.get('type'),
                'values': group1.get('values', []),
                'statistics': stats1
            },
            'group2': {
                'type': group2.get('type'),
                'values': group2.get('values', []),
                'statistics': stats2
            },
            'differences': {
                'count': {
                    'absolute': count_diff,
                    'percentage': round(count_diff_pct, 2)
                },
                'mean': {
                    'absolute': round(mean_diff, 2),
                    'percentage': round(mean_diff_pct, 2)
                },
                'median': {
                    'absolute': round(median_diff, 2),
                    'percentage': round(median_diff_pct, 2)
                },
                'max': {
                    'absolute': round(max_diff, 2),
                    'percentage': round(max_diff_pct, 2)
                }
            }
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _get_group_stats(group):
    """Helper function to get statistics for a group"""
    group_type = group.get('type')
    values = group.get('values', [])
    
    if not group_type or not values:
        return {'error': 'Group type and values required'}
    
    if group_type not in ['borough', 'category']:
        return {'error': 'Group type must be "borough" or "category"'}
    
    # Build query
    query = db.session.query(
        func.count(Location.id).label('count'),
        func.avg(AggregatedCount.avg_recent_count).label('mean'),
        func.min(AggregatedCount.avg_recent_count).label('min'),
        func.max(AggregatedCount.avg_recent_count).label('max'),
        func.stddev(AggregatedCount.avg_recent_count).label('std_dev')
    ).join(
        AggregatedCount, Location.id == AggregatedCount.location_id
    )
    
    # Apply filter based on group type
    if group_type == 'borough':
        query = query.filter(Location.borough.in_(values))
    elif group_type == 'category':
        query = query.filter(Location.category.in_(values))
    
    result = query.first()
    
    # Calculate median
    median_query = db.session.query(
        func.percentile_cont(0.5).within_group(
            AggregatedCount.avg_recent_count
        ).label('median')
    ).join(
        Location, AggregatedCount.location_id == Location.id
    )
    
    if group_type == 'borough':
        median_query = median_query.filter(Location.borough.in_(values))
    elif group_type == 'category':
        median_query = median_query.filter(Location.category.in_(values))
    
    median = median_query.scalar()
    
    return {
        'count': result.count or 0,
        'mean': float(result.mean) if result.mean else 0,
        'median': float(median) if median else 0,
        'min': float(result.min) if result.min else 0,
        'max': float(result.max) if result.max else 0,
        'std_dev': float(result.std_dev) if result.std_dev else 0
    }

