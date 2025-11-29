"""
Location model
"""
from app import db
from geoalchemy2 import Geometry
from sqlalchemy import Column, Integer, String, DateTime, func
from sqlalchemy.orm import relationship


class Location(db.Model):
    """Location model for pedestrian count locations"""
    __tablename__ = 'locations'
    
    id = Column(Integer, primary_key=True)
    objectid = Column(Integer, unique=True, nullable=False, index=True)
    loc_id = Column(Integer, index=True)
    borough = Column(String(50), index=True)
    street_name_clean = Column(String(255))
    street_clean = Column(String(255))
    category = Column(String(50), index=True)
    segmentid = Column(Integer, index=True)
    geometry = Column(Geometry('POINT', srid=4326), index=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    counts = relationship('PedestrianCount', back_populates='location', cascade='all, delete-orphan')
    aggregated_count = relationship('AggregatedCount', back_populates='location', uselist=False, cascade='all, delete-orphan')
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'objectid': self.objectid,
            'loc_id': self.loc_id,
            'borough': self.borough,
            'street_name_clean': self.street_name_clean,
            'street_clean': self.street_clean,
            'category': self.category,
            'segmentid': self.segmentid,
            'geometry': self.geometry,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def to_geojson(self):
        """Convert to GeoJSON Feature"""
        if self.geometry is None:
            return None
        
        # Extract coordinates from PostGIS geometry
        from sqlalchemy import text
        result = db.session.execute(
            text("SELECT ST_AsGeoJSON(:geom) as geojson"),
            {'geom': self.geometry}
        ).scalar()
        
        import json
        geom_data = json.loads(result)
        
        return {
            'type': 'Feature',
            'geometry': geom_data,
            'properties': {
                'id': self.id,
                'objectid': self.objectid,
                'loc_id': self.loc_id,
                'borough': self.borough,
                'street_name_clean': self.street_name_clean,
                'street_clean': self.street_clean,
                'category': self.category,
                'segmentid': self.segmentid,
                'avg_recent_count': self.aggregated_count.avg_recent_count if self.aggregated_count else None
            }
        }
    
    def __repr__(self):
        return f'<Location {self.objectid}: {self.street_name_clean}>'

