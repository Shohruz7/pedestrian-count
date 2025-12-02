"""
Pedestrian count models
"""
from app import db
from sqlalchemy import Column, Integer, Date, Numeric, String, DateTime, ForeignKey, CheckConstraint, func, UniqueConstraint
from sqlalchemy.orm import relationship


class PedestrianCount(db.Model):
    """Time series pedestrian count data"""
    __tablename__ = 'pedestrian_counts'
    
    id = Column(Integer, primary_key=True)
    location_id = Column(Integer, ForeignKey('locations.id', ondelete='CASCADE'), nullable=False, index=True)
    count_date = Column(Date, nullable=False, index=True)
    period = Column(String(10), nullable=False, index=True)  # AM, PM, MD
    count_value = Column(Numeric(10, 2))
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    location = relationship('Location', back_populates='counts')
    
    # Constraints
    __table_args__ = (
        CheckConstraint("period IN ('AM', 'PM', 'MD')", name='check_period'),
        UniqueConstraint('location_id', 'count_date', 'period', name='unique_location_date_period'),
    )
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'location_id': self.location_id,
            'count_date': self.count_date.isoformat() if self.count_date else None,
            'period': self.period,
            'count_value': float(self.count_value) if self.count_value else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def __repr__(self):
        return f'<PedestrianCount {self.location_id}: {self.count_date} {self.period}>'


class AggregatedCount(db.Model):
    """Pre-computed aggregated counts for performance"""
    __tablename__ = 'aggregated_counts'
    
    id = Column(Integer, primary_key=True)
    location_id = Column(Integer, ForeignKey('locations.id', ondelete='CASCADE'), unique=True, nullable=False, index=True)
    avg_recent_count = Column(Numeric(10, 2))
    min_count = Column(Numeric(10, 2))
    max_count = Column(Numeric(10, 2))
    std_dev = Column(Numeric(10, 2))
    count_records = Column(Integer, default=0)
    last_updated = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    location = relationship('Location', back_populates='aggregated_count')
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'location_id': self.location_id,
            'avg_recent_count': float(self.avg_recent_count) if self.avg_recent_count else None,
            'min_count': float(self.min_count) if self.min_count else None,
            'max_count': float(self.max_count) if self.max_count else None,
            'std_dev': float(self.std_dev) if self.std_dev else None,
            'count_records': self.count_records,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }
    
    def __repr__(self):
        return f'<AggregatedCount {self.location_id}: avg={self.avg_recent_count}>'


