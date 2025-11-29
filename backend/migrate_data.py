"""
Data migration script: CSV/GeoJSON to PostgreSQL
Run this script to populate the database from CSV and GeoJSON files
"""
import os
import sys
import pandas as pd
import geopandas as gpd
from datetime import datetime
import re
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db
from app.models import Location, PedestrianCount, AggregatedCount, DemandSegment

load_dotenv()


def parse_time_series_columns(df_raw):
    """Parse time series columns from Pedestrian_Counts.csv"""
    time_cols = [c for c in df_raw.columns if re.match(r'^(May|Sept|Oct|June|Apr|Mar|Feb|Jan|Nov|Dec)\d{2}_(AM|PM|MD)', c)]
    
    time_data = []
    for col in time_cols:
        match = re.match(r'^(\w+)(\d{2})_(AM|PM|MD)$', col)
        if match:
            month_str, year_str, period = match.groups()
            year = 2000 + int(year_str)
            month_map = {
                'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6,
                'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sept': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
            }
            month = month_map.get(month_str, 1)
            date = datetime(year, month, 15)
            time_data.append({
                'column': col,
                'date': date,
                'period': period,
                'year': year,
                'month': month
            })
    
    return time_data, time_cols


def load_locations(session, csv_path, geo_path):
    """Load locations from CSV and GeoJSON"""
    print("Loading locations...")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    df['avg_recent_count'] = pd.to_numeric(df['avg_recent_count'], errors='coerce')
    
    # Standardize borough names
    borough_mapping = {
        'Bronx': 'The Bronx',
        'Staten Isla': 'Staten Island',
        'Staten Island': 'Staten Island',
        'Brooklyn': 'Brooklyn',
        'Queens': 'Queens',
        'Manhattan': 'Manhattan',
        'East River Bridges': 'Bridges',
        'Harlem River Bridges': 'Bridges'
    }
    df['Borough'] = df['Borough'].replace(borough_mapping)
    valid_boroughs = ['The Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island', 'Bridges']
    df = df[df['Borough'].isin(valid_boroughs)].copy()
    
    # Read GeoJSON if available
    gdf = None
    if os.path.exists(geo_path):
        try:
            gdf = gpd.read_file(geo_path)
            gdf['avg_recent_count'] = pd.to_numeric(gdf['avg_recent_count'], errors='coerce')
            if gdf.crs != 'EPSG:4326':
                gdf = gdf.to_crs('EPSG:4326')
            if 'Borough' in gdf.columns:
                gdf['Borough'] = gdf['Borough'].replace(borough_mapping)
                gdf = gdf[gdf['Borough'].isin(valid_boroughs)].copy()
        except Exception as e:
            print(f"Warning: Could not load GeoJSON: {e}")
    
    # Merge data
    if gdf is not None and not gdf.empty:
        # Merge on OBJECTID
        merged = gdf.merge(df, on='OBJECTID', how='inner', suffixes=('', '_csv'))
    else:
        # Use CSV only, create geometry from lat/lon if available
        merged = df.copy()
        if 'latitude' in df.columns and 'longitude' in df.columns:
            from shapely.geometry import Point
            merged['geometry'] = merged.apply(
                lambda row: Point(row['longitude'], row['latitude']) if pd.notna(row['longitude']) and pd.notna(row['latitude']) else None,
                axis=1
            )
    
    # Insert locations
    locations_created = 0
    for idx, row in merged.iterrows():
        try:
            # Check if location already exists
            existing = session.query(Location).filter_by(objectid=int(row['OBJECTID'])).first()
            if existing:
                continue
            
            location = Location(
                objectid=int(row['OBJECTID']),
                loc_id=int(row['Loc']) if pd.notna(row['Loc']) else None,
                borough=row['Borough'] if pd.notna(row['Borough']) else None,
                street_name_clean=row['Street_Nam_clean'] if pd.notna(row.get('Street_Nam_clean')) else None,
                street_clean=row['street_clean'] if pd.notna(row.get('street_clean')) else None,
                category=row['Category'] if pd.notna(row['Category']) else None,
                segmentid=int(row['segmentid']) if pd.notna(row.get('segmentid')) else None,
                geometry=row['geometry'] if 'geometry' in row and pd.notna(row['geometry']) else None
            )
            
            session.add(location)
            locations_created += 1
            
            # Create aggregated count if avg_recent_count exists
            if pd.notna(row.get('avg_recent_count')):
                aggregated = AggregatedCount(
                    location=location,
                    avg_recent_count=float(row['avg_recent_count']),
                    count_records=1
                )
                session.add(aggregated)
        
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    session.commit()
    print(f"Created {locations_created} locations")
    return locations_created


def load_time_series(session, raw_path):
    """Load time series data from Pedestrian_Counts.csv"""
    print("Loading time series data...")
    
    if not os.path.exists(raw_path):
        print(f"Warning: {raw_path} not found. Skipping time series data.")
        return 0
    
    # Read raw data
    df_raw = pd.read_csv(raw_path, dtype=str)
    
    # Parse time series columns
    time_data, time_cols = parse_time_series_columns(df_raw)
    if not time_cols:
        print("No time series columns found. Skipping time series data.")
        return 0
    
    # Get all locations
    locations = session.query(Location).all()
    location_map = {loc.objectid: loc for loc in locations}
    
    counts_created = 0
    for idx, row in df_raw.iterrows():
        try:
            objectid = int(row['OBJECTID'])
            if objectid not in location_map:
                continue
            
            location = location_map[objectid]
            
            # Process each time series column
            for time_info in time_data:
                col = time_info['column']
                if col not in row or pd.isna(row[col]):
                    continue
                
                try:
                    count_value = float(row[col])
                    if pd.isna(count_value) or count_value < 0:
                        continue
                    
                    # Check if count already exists
                    existing = session.query(PedestrianCount).filter_by(
                        location_id=location.id,
                        count_date=time_info['date'].date(),
                        period=time_info['period']
                    ).first()
                    
                    if existing:
                        continue
                    
                    count = PedestrianCount(
                        location_id=location.id,
                        count_date=time_info['date'].date(),
                        period=time_info['period'],
                        count_value=count_value
                    )
                    session.add(count)
                    counts_created += 1
                
                except (ValueError, TypeError):
                    continue
        
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    session.commit()
    print(f"Created {counts_created} pedestrian count records")
    return counts_created


def load_demand_segments(session, demand_path):
    """Load demand segments from Pedestrian_Demand.csv"""
    print("Loading demand segments...")
    
    if not os.path.exists(demand_path):
        print(f"Warning: {demand_path} not found. Skipping demand segments.")
        return 0
    
    # Read CSV
    df = pd.read_csv(demand_path, dtype=str)
    
    # Parse geometry if available
    if 'the_geom' in df.columns:
        from shapely import wkt
        df['geometry'] = df['the_geom'].apply(
            lambda x: wkt.loads(x) if pd.notna(x) and x else None
        )
    
    segments_created = 0
    for idx, row in df.iterrows():
        try:
            # Check if segment already exists
            if pd.notna(row.get('segmentid')):
                existing = session.query(DemandSegment).filter_by(segmentid=int(row['segmentid'])).first()
                if existing:
                    continue
            
            segment = DemandSegment(
                segmentid=int(row['segmentid']) if pd.notna(row.get('segmentid')) else None,
                street=row['street'] if pd.notna(row.get('street')) else None,
                boro_code=int(row['BoroCode']) if pd.notna(row.get('BoroCode')) else None,
                boro_name=row['BoroName'] if pd.notna(row.get('BoroName')) else None,
                boro_cd=int(row['BoroCD']) if pd.notna(row.get('BoroCD')) else None,
                coun_dist=int(row['CounDist']) if pd.notna(row.get('CounDist')) else None,
                assem_dist=int(row['AssemDist']) if pd.notna(row.get('AssemDist')) else None,
                st_sen_dist=int(row['StSenDist']) if pd.notna(row.get('StSenDist')) else None,
                cong_dist=int(row['CongDist']) if pd.notna(row.get('CongDist')) else None,
                rank=int(row['Rank']) if pd.notna(row.get('Rank')) else None,
                pmp_id=row['PMP_ID'] if pd.notna(row.get('PMP_ID')) else None,
                nta2020=row['NTA2020'] if pd.notna(row.get('NTA2020')) else None,
                boro=row['Boro'] if pd.notna(row.get('Boro')) else None,
                category=row['Category'] if pd.notna(row.get('Category')) else None,
                nta_name=row['NTAName'] if pd.notna(row.get('NTAName')) else None,
                fema_fldz=row['FEMAFldz'] if pd.notna(row.get('FEMAFldz')) else None,
                fema_fldt=row['FEMAFldT'] if pd.notna(row.get('FEMAFldT')) else None,
                hrc_evac=row['HrcEvac'] if pd.notna(row.get('HrcEvac')) else None,
                shape_leng=float(row['SHAPE_Leng']) if pd.notna(row.get('SHAPE_Leng')) else None,
                geometry=row['geometry'] if 'geometry' in row and pd.notna(row['geometry']) else None
            )
            
            session.add(segment)
            segments_created += 1
        
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    session.commit()
    print(f"Created {segments_created} demand segments")
    return segments_created


def main():
    """Main migration function"""
    app = create_app()
    
    with app.app_context():
        # Create tables if they don't exist
        print("Creating database tables...")
        db.create_all()
        
        # Get data paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, 'data_clean', 'pedestrian_combined.csv')
        geo_path = os.path.join(base_dir, 'data_clean', 'pedestrian_combined.geojson')
        raw_path = os.path.join(base_dir, 'data-raw', 'Pedestrian_Counts.csv')
        demand_path = os.path.join(base_dir, 'data-raw', 'Pedestrian_Demand.csv')
        
        # Load data
        session = db.session
        
        try:
            locations_count = load_locations(session, csv_path, geo_path)
            time_series_count = load_time_series(session, raw_path)
            demand_count = load_demand_segments(session, demand_path)
            
            print("\n" + "="*50)
            print("Migration Summary:")
            print(f"  Locations: {locations_count}")
            print(f"  Time series records: {time_series_count}")
            print(f"  Demand segments: {demand_count}")
            print("="*50)
        
        except Exception as e:
            session.rollback()
            print(f"Error during migration: {e}")
            raise


if __name__ == '__main__':
    main()

