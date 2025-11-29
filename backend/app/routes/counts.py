"""
Time series count API routes
"""
from flask import Blueprint, request, jsonify
from datetime import datetime
from sqlalchemy import and_, func
from app import db
from app.models import Location, PedestrianCount

bp = Blueprint('counts', __name__)


@bp.route('/time-series/<int:location_id>', methods=['GET'])
def get_time_series(location_id):
    """Get time series data for a specific location"""
    try:
        location = db.session.query(Location).filter_by(id=location_id).first()
        if not location:
            return jsonify({'error': 'Location not found'}), 404
        
        # Get query parameters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        periods = request.args.getlist('period[]')  # ['AM', 'PM', 'MD']
        
        # Build query
        query = db.session.query(PedestrianCount).filter_by(location_id=location_id)
        
        if start_date:
            try:
                start = datetime.strptime(start_date, '%Y-%m-%d').date()
                query = query.filter(PedestrianCount.count_date >= start)
            except ValueError:
                return jsonify({'error': 'Invalid start_date format. Use YYYY-MM-DD'}), 400
        
        if end_date:
            try:
                end = datetime.strptime(end_date, '%Y-%m-%d').date()
                query = query.filter(PedestrianCount.count_date <= end)
            except ValueError:
                return jsonify({'error': 'Invalid end_date format. Use YYYY-MM-DD'}), 400
        
        if periods:
            valid_periods = ['AM', 'PM', 'MD']
            periods = [p for p in periods if p in valid_periods]
            if periods:
                query = query.filter(PedestrianCount.period.in_(periods))
        
        # Order by date and period
        counts = query.order_by(
            PedestrianCount.count_date,
            PedestrianCount.period
        ).all()
        
        return jsonify({
            'location_id': location_id,
            'location': location.to_dict(),
            'counts': [count.to_dict() for count in counts],
            'total_records': len(counts)
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/time-series/aggregate', methods=['GET'])
def get_aggregate_time_series():
    """Get aggregated time series data across multiple locations"""
    try:
        # Get query parameters
        location_ids = request.args.getlist('location_id[]', type=int)
        periods = request.args.getlist('period[]')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Build query
        query = db.session.query(
            PedestrianCount.count_date,
            PedestrianCount.period,
            func.avg(PedestrianCount.count_value).label('avg_count'),
            func.min(PedestrianCount.count_value).label('min_count'),
            func.max(PedestrianCount.count_value).label('max_count'),
            func.count(PedestrianCount.id).label('record_count')
        )
        
        if location_ids:
            query = query.filter(PedestrianCount.location_id.in_(location_ids))
        
        if periods:
            valid_periods = ['AM', 'PM', 'MD']
            periods = [p for p in periods if p in valid_periods]
            if periods:
                query = query.filter(PedestrianCount.period.in_(periods))
        
        if start_date:
            try:
                start = datetime.strptime(start_date, '%Y-%m-%d').date()
                query = query.filter(PedestrianCount.count_date >= start)
            except ValueError:
                return jsonify({'error': 'Invalid start_date format. Use YYYY-MM-DD'}), 400
        
        if end_date:
            try:
                end = datetime.strptime(end_date, '%Y-%m-%d').date()
                query = query.filter(PedestrianCount.count_date <= end)
            except ValueError:
                return jsonify({'error': 'Invalid end_date format. Use YYYY-MM-DD'}), 400
        
        # Group by date and period
        results = query.group_by(
            PedestrianCount.count_date,
            PedestrianCount.period
        ).order_by(
            PedestrianCount.count_date,
            PedestrianCount.period
        ).all()
        
        aggregated = []
        for row in results:
            aggregated.append({
                'date': row.count_date.isoformat() if row.count_date else None,
                'period': row.period,
                'avg_count': float(row.avg_count) if row.avg_count else 0,
                'min_count': float(row.min_count) if row.min_count else 0,
                'max_count': float(row.max_count) if row.max_count else 0,
                'record_count': row.record_count
            })
        
        return jsonify({
            'aggregated_data': aggregated,
            'total_records': len(aggregated)
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

