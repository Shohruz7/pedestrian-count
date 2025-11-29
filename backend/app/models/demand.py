"""
Demand segment model
"""
from app import db
from geoalchemy2 import Geometry
from sqlalchemy import Column, Integer, String, Numeric, DateTime, func


class DemandSegment(db.Model):
    """Demand segment model from Pedestrian_Demand.csv"""
    __tablename__ = 'demand_segments'
    
    id = Column(Integer, primary_key=True)
    segmentid = Column(Integer, unique=True, index=True)
    street = Column(String(255))
    boro_code = Column(Integer)
    boro_name = Column(String(50))
    boro_cd = Column(Integer)
    coun_dist = Column(Integer)
    assem_dist = Column(Integer)
    st_sen_dist = Column(Integer)
    cong_dist = Column(Integer)
    rank = Column(Integer)
    pmp_id = Column(String(50))
    nta2020 = Column(String(50))
    boro = Column(String(50), index=True)
    category = Column(String(50), index=True)
    nta_name = Column(String(255))
    fema_fldz = Column(String(50))
    fema_fldt = Column(String(255))
    hrc_evac = Column(String(50))
    shape_leng = Column(Numeric(10, 6))
    geometry = Column(Geometry('LINESTRING', srid=4326), index=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'segmentid': self.segmentid,
            'street': self.street,
            'boro_code': self.boro_code,
            'boro_name': self.boro_name,
            'boro': self.boro,
            'category': self.category,
            'rank': self.rank,
            'shape_leng': float(self.shape_leng) if self.shape_leng else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def to_geojson(self):
        """Convert to GeoJSON Feature"""
        if self.geometry is None:
            return None
        
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
            'properties': self.to_dict()
        }
    
    def __repr__(self):
        return f'<DemandSegment {self.segmentid}: {self.street}>'

