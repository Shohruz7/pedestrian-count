-- PostgreSQL Database Schema for Pedestrian Count Dashboard
-- Requires PostGIS extension for geospatial data

CREATE EXTENSION IF NOT EXISTS postgis;

-- ============================================================================
-- LOCATIONS TABLE
-- Main table for pedestrian count locations
-- ============================================================================
CREATE TABLE IF NOT EXISTS locations (
    id SERIAL PRIMARY KEY,
    objectid INTEGER UNIQUE NOT NULL,
    loc_id INTEGER,
    borough VARCHAR(50),
    street_name_clean VARCHAR(255),
    street_clean VARCHAR(255),
    category VARCHAR(50),
    segmentid INTEGER,
    geometry GEOMETRY(POINT, 4326),  -- PostGIS geometry (WGS84)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create spatial index for geometry column (GIST index for PostGIS)
CREATE INDEX IF NOT EXISTS idx_locations_geometry ON locations USING GIST (geometry);

-- Create indexes for common filter columns
CREATE INDEX IF NOT EXISTS idx_locations_borough ON locations (borough);
CREATE INDEX IF NOT EXISTS idx_locations_category ON locations (category);
CREATE INDEX IF NOT EXISTS idx_locations_segmentid ON locations (segmentid);
CREATE INDEX IF NOT EXISTS idx_locations_loc_id ON locations (loc_id);

-- ============================================================================
-- PEDESTRIAN_COUNTS TABLE
-- Time series data for pedestrian counts
-- ============================================================================
CREATE TABLE IF NOT EXISTS pedestrian_counts (
    id SERIAL PRIMARY KEY,
    location_id INTEGER NOT NULL REFERENCES locations(id) ON DELETE CASCADE,
    count_date DATE NOT NULL,
    period VARCHAR(10) NOT NULL CHECK (period IN ('AM', 'PM', 'MD')),
    count_value NUMERIC(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(location_id, count_date, period)
);

-- Create indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_pedestrian_counts_location_id ON pedestrian_counts (location_id);
CREATE INDEX IF NOT EXISTS idx_pedestrian_counts_date ON pedestrian_counts (count_date);
CREATE INDEX IF NOT EXISTS idx_pedestrian_counts_period ON pedestrian_counts (period);
CREATE INDEX IF NOT EXISTS idx_pedestrian_counts_location_date ON pedestrian_counts (location_id, count_date, period);

-- ============================================================================
-- AGGREGATED_COUNTS TABLE
-- Pre-computed aggregations for performance
-- ============================================================================
CREATE TABLE IF NOT EXISTS aggregated_counts (
    id SERIAL PRIMARY KEY,
    location_id INTEGER UNIQUE NOT NULL REFERENCES locations(id) ON DELETE CASCADE,
    avg_recent_count NUMERIC(10, 2),
    min_count NUMERIC(10, 2),
    max_count NUMERIC(10, 2),
    std_dev NUMERIC(10, 2),
    count_records INTEGER DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_aggregated_counts_location_id ON aggregated_counts (location_id);

-- ============================================================================
-- DEMAND_SEGMENTS TABLE
-- Data from Pedestrian_Demand.csv
-- ============================================================================
CREATE TABLE IF NOT EXISTS demand_segments (
    id SERIAL PRIMARY KEY,
    segmentid INTEGER UNIQUE,
    street VARCHAR(255),
    boro_code INTEGER,
    boro_name VARCHAR(50),
    boro_cd INTEGER,
    coun_dist INTEGER,
    assem_dist INTEGER,
    st_sen_dist INTEGER,
    cong_dist INTEGER,
    rank INTEGER,
    pmp_id VARCHAR(50),
    nta2020 VARCHAR(50),
    boro VARCHAR(50),
    category VARCHAR(50),
    nta_name VARCHAR(255),
    fema_fldz VARCHAR(50),
    fema_fldt VARCHAR(255),
    hrc_evac VARCHAR(50),
    shape_leng NUMERIC(10, 6),
    geometry GEOMETRY(LINESTRING, 4326),  -- PostGIS geometry for line segments
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create spatial index for demand segments
CREATE INDEX IF NOT EXISTS idx_demand_segments_geometry ON demand_segments USING GIST (geometry);
CREATE INDEX IF NOT EXISTS idx_demand_segments_segmentid ON demand_segments (segmentid);
CREATE INDEX IF NOT EXISTS idx_demand_segments_boro ON demand_segments (boro);
CREATE INDEX IF NOT EXISTS idx_demand_segments_category ON demand_segments (category);

-- ============================================================================
-- FUNCTIONS AND TRIGGERS
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for locations table
CREATE TRIGGER update_locations_updated_at BEFORE UPDATE ON locations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Trigger for demand_segments table
CREATE TRIGGER update_demand_segments_updated_at BEFORE UPDATE ON demand_segments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to update aggregated_counts when pedestrian_counts change
CREATE OR REPLACE FUNCTION update_aggregated_counts()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO aggregated_counts (location_id, avg_recent_count, min_count, max_count, std_dev, count_records, last_updated)
    SELECT 
        location_id,
        AVG(count_value) as avg_recent_count,
        MIN(count_value) as min_count,
        MAX(count_value) as max_count,
        STDDEV(count_value) as std_dev,
        COUNT(*) as count_records,
        CURRENT_TIMESTAMP
    FROM pedestrian_counts
    WHERE location_id = COALESCE(NEW.location_id, OLD.location_id)
    GROUP BY location_id
    ON CONFLICT (location_id) 
    DO UPDATE SET
        avg_recent_count = EXCLUDED.avg_recent_count,
        min_count = EXCLUDED.min_count,
        max_count = EXCLUDED.max_count,
        std_dev = EXCLUDED.std_dev,
        count_records = EXCLUDED.count_records,
        last_updated = CURRENT_TIMESTAMP;
    
    RETURN COALESCE(NEW, OLD);
END;
$$ language 'plpgsql';

-- Trigger to auto-update aggregated_counts
CREATE TRIGGER update_aggregated_on_count_change
    AFTER INSERT OR UPDATE OR DELETE ON pedestrian_counts
    FOR EACH ROW EXECUTE FUNCTION update_aggregated_counts();

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- View for locations with aggregated counts
CREATE OR REPLACE VIEW locations_with_counts AS
SELECT 
    l.*,
    COALESCE(a.avg_recent_count, 0) as avg_recent_count,
    COALESCE(a.min_count, 0) as min_count,
    COALESCE(a.max_count, 0) as max_count,
    COALESCE(a.std_dev, 0) as std_dev,
    COALESCE(a.count_records, 0) as count_records
FROM locations l
LEFT JOIN aggregated_counts a ON l.id = a.location_id;

-- View for borough statistics
CREATE OR REPLACE VIEW borough_statistics AS
SELECT 
    borough,
    COUNT(*) as location_count,
    AVG(avg_recent_count) as avg_count,
    MIN(avg_recent_count) as min_count,
    MAX(avg_recent_count) as max_count,
    STDDEV(avg_recent_count) as std_dev,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY avg_recent_count) as median_count
FROM locations_with_counts
WHERE borough IS NOT NULL
GROUP BY borough;

-- View for category statistics
CREATE OR REPLACE VIEW category_statistics AS
SELECT 
    category,
    COUNT(*) as location_count,
    AVG(avg_recent_count) as avg_count,
    MIN(avg_recent_count) as min_count,
    MAX(avg_recent_count) as max_count,
    STDDEV(avg_recent_count) as std_dev,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY avg_recent_count) as median_count
FROM locations_with_counts
WHERE category IS NOT NULL
GROUP BY category;


