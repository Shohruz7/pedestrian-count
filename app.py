"""
Pedestrian Count Dashboard
Streamlit app for visualizing NYC pedestrian count data
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import re

# Page configuration
st.set_page_config(
    page_title="NYC Pedestrian Count Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    """Load the cleaned pedestrian count data"""
    csv_path = "data_clean/pedestrian_combined.csv"
    geo_path = "data_clean/pedestrian_combined.geojson"
    
    df = pd.read_csv(csv_path)
    df['avg_recent_count'] = pd.to_numeric(df['avg_recent_count'], errors='coerce')
    
    # Try to load GeoJSON for mapping
    gdf = None
    if os.path.exists(geo_path):
        try:
            gdf = gpd.read_file(geo_path)
            gdf['avg_recent_count'] = pd.to_numeric(gdf['avg_recent_count'], errors='coerce')
            # Ensure CRS is WGS84 for folium
            if gdf.crs != 'EPSG:4326':
                gdf = gdf.to_crs('EPSG:4326')
            # Use geometry column (it should be named 'geometry' after GeoPandas read)
            if 'geometry' not in gdf.columns and 'the_geom' in gdf.columns:
                gdf = gdf.set_geometry('the_geom')
            # Select only needed columns for mapping
            cols_needed = ['OBJECTID', 'Loc', 'Borough', 'Street_Nam_clean', 
                          'avg_recent_count', 'Category', 'geometry']
            gdf = gdf[[c for c in cols_needed if c in gdf.columns]]
        except Exception as e:
            st.warning(f"Could not load GeoJSON: {e}. Maps will use CSV coordinates if available.")
    
    return df, gdf

@st.cache_data
def load_time_series_data():
    """Load raw counts data for time series analysis"""
    try:
        raw_path = "data-raw/Pedestrian_Counts.csv"
        if os.path.exists(raw_path):
            df_raw = pd.read_csv(raw_path, dtype=str)
            
            # Identify time series columns (format: MonthYY_Period, e.g., May07_AM)
            time_cols = [c for c in df_raw.columns if re.match(r'^(May|Sept|Oct|June|Apr|Mar|Feb|Jan|Nov|Dec)\d{2}_(AM|PM|MD)', c)]
            
            if len(time_cols) > 0:
                # Parse dates from column names
                time_data = []
                for col in time_cols:
                    match = re.match(r'^(\w+)(\d{2})_(AM|PM|MD)$', col)
                    if match:
                        month_str, year_str, period = match.groups()
                        year = 2000 + int(year_str)
                        # Convert month name to number
                        month_map = {
                            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6,
                            'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sept': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                        }
                        month = month_map.get(month_str, 1)
                        # Use 15th as approximate day
                        date = datetime(year, month, 15)
                        time_data.append({
                            'column': col,
                            'date': date,
                            'period': period,
                            'year': year,
                            'month': month
                        })
                
                time_df = pd.DataFrame(time_data).sort_values('date')
                return df_raw, time_df, time_cols
    except Exception as e:
        pass
    
    return None, None, None

# Load data
df, gdf = load_data()
df_raw, time_df, time_cols = load_time_series_data()

# Standardize borough names and keep bridges separate
# Map old names to new standardized names
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
# Apply mapping to DataFrame
df['Borough'] = df['Borough'].replace(borough_mapping)

# Keep all 5 boroughs + Bridges (6 total categories)
valid_boroughs = ['The Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island', 'Bridges']
df = df[df['Borough'].isin(valid_boroughs)].copy()

# Apply same mapping to GeoDataFrame if it exists
if gdf is not None and not gdf.empty:
    if 'Borough' in gdf.columns:
        gdf['Borough'] = gdf['Borough'].replace(borough_mapping)
        gdf = gdf[gdf['Borough'].isin(valid_boroughs)].copy()

# Title and description
st.title("NYC Pedestrian Count Dashboard")
st.markdown("""
This dashboard visualizes pedestrian count data from NYC DOT's Bi-Annual Pedestrian Counts 
and Pedestrian Mobility Plan Pedestrian Demand Map datasets.
""")

# Sidebar info
with st.sidebar:
    st.header("Dataset Info")
    st.metric("Total Locations", len(df))
    # Count all boroughs including Bridges (6 total: 5 boroughs + Bridges)
    all_boroughs = ['The Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island', 'Bridges']
    borough_count = len([b for b in df['Borough'].unique() if b in all_boroughs])
    st.metric("Boroughs & Bridges", borough_count)
    st.metric("Categories", 5)  # All 5 categories: Global, Regional, Neighborhood, Community, Baseline
    
    st.divider()
    st.markdown("### Category Distribution")
    # Show all 5 categories (even if some have 0 locations)
    all_categories = ['Global', 'Regional', 'Neighborhood', 'Community', 'Baseline']
    cat_counts = df['Category'].value_counts()
    for cat in all_categories:
        count = cat_counts.get(cat, 0)
        st.text(f"{cat}: {count}")

# Initialize filter options (only valid boroughs - already filtered above)
boroughs = sorted(df['Borough'].unique())
categories = sorted(df['Category'].unique())
min_count = float(df['avg_recent_count'].min())
max_count = float(df['avg_recent_count'].max())

# Initialize session state for filters if not exists
if 'selected_boroughs' not in st.session_state:
    st.session_state.selected_boroughs = boroughs
if 'selected_categories' not in st.session_state:
    st.session_state.selected_categories = categories
if 'count_range' not in st.session_state:
    st.session_state.count_range = (min_count, max_count)
if 'search_results' not in st.session_state:
    st.session_state.search_results = None

# Apply filters to data (using session state values)
if len(st.session_state.selected_boroughs) > 0 and len(st.session_state.selected_categories) > 0:
    filtered_df = df[
        (df['Borough'].isin(st.session_state.selected_boroughs)) &
        (df['Category'].isin(st.session_state.selected_categories)) &
        (df['avg_recent_count'] >= st.session_state.count_range[0]) &
        (df['avg_recent_count'] <= st.session_state.count_range[1])
    ]
    
    # Apply filters to GeoDataFrame if available
    if gdf is not None:
        filtered_gdf = gdf[
            (gdf['Borough'].isin(st.session_state.selected_boroughs)) &
            (gdf['Category'].isin(st.session_state.selected_categories)) &
            (gdf['avg_recent_count'] >= st.session_state.count_range[0]) &
            (gdf['avg_recent_count'] <= st.session_state.count_range[1])
        ]
    else:
        filtered_gdf = None
else:
    # If nothing selected, show empty results but keep gdf structure
    filtered_df = pd.DataFrame()
    filtered_gdf = gpd.GeoDataFrame() if gdf is not None else None

# Create tabs for different features
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Map & Filters", 
    "Comparison", 
    "Statistics", 
    "Time Series", 
    "Top Sites",
    "Export"
])

# TAB 1: Map & Filters
with tab1:
    st.header("Interactive Map")
    
    # Show map based on filtered data
    if filtered_gdf is not None and len(filtered_gdf) > 0:
        # Create a folium map centered on NYC
        m = folium.Map(
            location=[40.7128, -73.9352],  # NYC coordinates
            zoom_start=11,
            tiles='OpenStreetMap'
        )
        
        # Map options in sidebar
        st.sidebar.divider()
        st.sidebar.subheader("Map Options")
        use_clustering = st.sidebar.checkbox("Enable Map Clustering", value=True, 
                                             help="Group nearby markers together for better performance")
        
        # Highlight search results if available
        highlight_search = False
        if 'search_results' in st.session_state and st.session_state.search_results is not None:
            if isinstance(st.session_state.search_results, pd.DataFrame) and len(st.session_state.search_results) > 0:
                highlight_search = st.sidebar.checkbox("Highlight Search Results", value=True,
                                                       help="Highlight locations matching your search")
        
        if use_clustering:
            # Create marker clusters by category for better organization
            cluster_groups = {}
            color_map = {
                'Global': 'red',
                'Regional': 'orange',
                'Neighborhood': 'blue',
                'Community': 'green',
                'Baseline': 'gray'
            }
            
            for category in filtered_gdf['Category'].unique():
                cluster_groups[category] = MarkerCluster(
                    name=f'{category} Locations',
                    overlay=True
                ).add_to(m)
            
            # Add markers to appropriate clusters
            for idx, row in filtered_gdf.iterrows():
                if pd.notna(row['geometry']):
                    # Get coordinates
                    if row['geometry'].geom_type == 'Point':
                        coords = [row['geometry'].y, row['geometry'].x]
                    else:
                        coords = [row['geometry'].centroid.y, row['geometry'].centroid.x]
                    
                    color = color_map.get(row['Category'], 'gray')
                    
                    # Create popup text
                    popup_text = f"""
                    <div style="font-family: Arial, sans-serif; padding: 5px; background-color: white; color: black; 
                                border: 2px solid {color}; border-radius: 5px; min-width: 200px; max-width: 300px;">
                        <h4 style="margin: 0 0 8px 0; color: {color}; font-size: 14px; word-wrap: break-word;">
                            {row['Street_Nam_clean']}
                        </h4>
                        <p style="margin: 4px 0; font-size: 12px; color: black;">
                            <strong>Borough:</strong> {row['Borough']}<br>
                            <strong>Category:</strong> {row['Category']}<br>
                            <strong>Avg Count:</strong> {row['avg_recent_count']:,.0f}<br>
                            <strong>Location ID:</strong> {row['Loc']}
                        </p>
                    </div>
                    """
                    
                    # Check if this location is in search results
                    is_search_result = False
                    if highlight_search and 'search_results' in st.session_state and st.session_state.search_results is not None:
                        if isinstance(st.session_state.search_results, pd.DataFrame):
                            is_search_result = row['OBJECTID'] in st.session_state.search_results['OBJECTID'].values
                    
                    # Adjust marker style for search results
                    marker_color = 'yellow' if is_search_result else color
                    marker_fill_color = 'yellow' if is_search_result else color
                    marker_opacity = 0.9 if is_search_result else 0.6
                    marker_radius = 8 + (row['avg_recent_count'] / 1000) if is_search_result else 5 + (row['avg_recent_count'] / 1000)
                    
                    # Add marker to appropriate cluster
                    folium.CircleMarker(
                        location=coords,
                        radius=marker_radius,
                        popup=folium.Popup(popup_text, max_width=350),
                        tooltip=f"{row['Street_Nam_clean']} ({row['avg_recent_count']:,.0f})" + (" [SEARCH RESULT]" if is_search_result else ""),
                        color='black' if is_search_result else color,
                        weight=3 if is_search_result else 1,
                        fill=True,
                        fillColor=marker_fill_color,
                        fillOpacity=marker_opacity
                    ).add_to(cluster_groups[row['Category']])
        else:
            # Original non-clustered markers
            for idx, row in filtered_gdf.iterrows():
                if pd.notna(row['geometry']):
                    # Get coordinates
                    if row['geometry'].geom_type == 'Point':
                        coords = [row['geometry'].y, row['geometry'].x]
                    else:
                        coords = [row['geometry'].centroid.y, row['geometry'].centroid.x]
                    
                    color_map = {
                        'Global': 'red',
                        'Regional': 'orange',
                        'Neighborhood': 'blue',
                        'Community': 'green',
                        'Baseline': 'gray'
                    }
                    color = color_map.get(row['Category'], 'gray')
                    
                    # Create popup text
                    popup_text = f"""
                    <div style="font-family: Arial, sans-serif; padding: 5px; background-color: white; color: black; 
                                border: 2px solid {color}; border-radius: 5px; min-width: 200px; max-width: 300px;">
                        <h4 style="margin: 0 0 8px 0; color: {color}; font-size: 14px; word-wrap: break-word;">
                            {row['Street_Nam_clean']}
                        </h4>
                        <p style="margin: 4px 0; font-size: 12px; color: black;">
                            <strong>Borough:</strong> {row['Borough']}<br>
                            <strong>Category:</strong> {row['Category']}<br>
                            <strong>Avg Count:</strong> {row['avg_recent_count']:,.0f}<br>
                            <strong>Location ID:</strong> {row['Loc']}
                        </p>
                    </div>
                    """
                    
                    # Check if this location is in search results
                    is_search_result = False
                    if highlight_search and 'search_results' in st.session_state and st.session_state.search_results is not None:
                        if isinstance(st.session_state.search_results, pd.DataFrame):
                            is_search_result = row['OBJECTID'] in st.session_state.search_results['OBJECTID'].values
                    
                    # Adjust marker style for search results
                    marker_color = 'yellow' if is_search_result else color
                    marker_fill_color = 'yellow' if is_search_result else color
                    marker_opacity = 0.9 if is_search_result else 0.6
                    marker_radius = 8 + (row['avg_recent_count'] / 1000) if is_search_result else 5 + (row['avg_recent_count'] / 1000)
                    
                    # Add marker
                    folium.CircleMarker(
                        location=coords,
                        radius=marker_radius,
                        popup=folium.Popup(popup_text, max_width=350),
                        tooltip=f"{row['Street_Nam_clean']} ({row['avg_recent_count']:,.0f})" + (" [SEARCH RESULT]" if is_search_result else ""),
                        color='black' if is_search_result else color,
                        weight=3 if is_search_result else 1,
                        fill=True,
                        fillColor=marker_fill_color,
                        fillOpacity=marker_opacity
                    ).add_to(m)
        
        # Add legend (5 categories)
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 200px; 
                    background-color: white; z-index:9999; font-size:14px;
                    border:2px solid grey; padding: 10px; overflow: visible;">
        <p style="margin: 0 0 8px 0; color: black; font-weight: bold;">Category Legend</p>
        <p style="margin: 4px 0; color: black;"><i class="fa fa-circle" style="color:red"></i> Global</p>
        <p style="margin: 4px 0; color: black;"><i class="fa fa-circle" style="color:orange"></i> Regional</p>
        <p style="margin: 4px 0; color: black;"><i class="fa fa-circle" style="color:blue"></i> Neighborhood</p>
        <p style="margin: 4px 0; color: black;"><i class="fa fa-circle" style="color:green"></i> Community</p>
        <p style="margin: 4px 0; color: black;"><i class="fa fa-circle" style="color:gray"></i> Baseline</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Display map
        st_folium(m, width=1200, height=600)
        
        st.caption("Marker size represents average pedestrian count. Click markers for details.")
        
    elif filtered_gdf is not None:
        # Show empty map when no locations match filters or nothing selected
        m = folium.Map(
            location=[40.7128, -73.9352],  # NYC coordinates
            zoom_start=11,
            tiles='OpenStreetMap'
        )
        st_folium(m, width=1200, height=600)
        if len(st.session_state.selected_boroughs) == 0 or len(st.session_state.selected_categories) == 0:
            st.info("Select at least one borough and one category to see locations on the map.")
        else:
            st.info("No locations match the selected filters. Adjust filters below to see locations.")
    elif gdf is not None:
        # GeoJSON exists but no filtered results - show empty map
        m = folium.Map(
            location=[40.7128, -73.9352],  # NYC coordinates
            zoom_start=11,
            tiles='OpenStreetMap'
        )
        st_folium(m, width=1200, height=600)
        st.info("No locations match the selected filters. Adjust filters below to see locations.")
    else:
        st.warning("GeoJSON file not available. Please ensure data_clean/pedestrian_combined.geojson exists.")
        st.dataframe(filtered_df.head(10) if len(filtered_df) > 0 else df.head(10))
    
    # Filters section (placed below the map)
    st.divider()
    st.header("Filters")
    
    # Search functionality
    st.subheader("Search")
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        search_query = st.text_input(
            "Search by street name or location ID",
            value="",
            placeholder="e.g., 'Broadway' or '57'",
            key="search_input",
            help="Search for locations by street name or location number"
        )
    with search_col2:
        search_button = st.button("Search", type="primary", use_container_width=True)
    
    # Perform search
    if search_button or search_query:
        if search_query.strip():
            # Search by street name (case-insensitive)
            street_matches = df[df['Street_Nam_clean'].str.contains(search_query.upper(), case=False, na=False)]
            # Search by location ID
            try:
                loc_id = int(search_query)
                loc_matches = df[df['Loc'] == loc_id]
            except ValueError:
                loc_matches = pd.DataFrame()
            
            # Combine results
            search_results = pd.concat([street_matches, loc_matches]).drop_duplicates(subset=['OBJECTID'])
            st.session_state.search_results = search_results
            
            if len(search_results) > 0:
                st.success(f"Found {len(search_results)} location(s) matching '{search_query}'")
                # Auto-select found locations in filters
                if len(search_results) <= 20:  # Only auto-select if reasonable number
                    st.session_state.selected_boroughs = sorted(search_results['Borough'].unique().tolist())
                    st.session_state.selected_categories = sorted(search_results['Category'].unique().tolist())
                    # Adjust count range to include search results
                    min_search = float(search_results['avg_recent_count'].min())
                    max_search = float(search_results['avg_recent_count'].max())
                    current_min = st.session_state.count_range[0]
                    current_max = st.session_state.count_range[1]
                    st.session_state.count_range = (
                        min(current_min, min_search),
                        max(current_max, max_search)
                    )
            else:
                st.warning(f"No locations found matching '{search_query}'")
                st.session_state.search_results = None
        else:
            st.session_state.search_results = None
    
    # Clear filters button
    col_clear, col_spacer = st.columns([1, 5])
    with col_clear:
        if st.button("Clear All Filters", type="secondary", help="Reset all filters to show all locations"):
            st.session_state.selected_boroughs = sorted(df['Borough'].unique())
            st.session_state.selected_categories = sorted(df['Category'].unique())
            st.session_state.count_range = (float(df['avg_recent_count'].min()), float(df['avg_recent_count'].max()))
            st.session_state.search_results = None
            st.rerun()
    
    st.subheader("Filter Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Borough filter
        selected_boroughs = st.multiselect(
            "Select Borough(s)",
            options=boroughs,
            default=st.session_state.selected_boroughs,
            help="Select one or more boroughs to filter locations",
            key="borough_filter"
        )
        st.session_state.selected_boroughs = selected_boroughs
    
    with col2:
        # Category filter
        selected_categories = st.multiselect(
            "Select Category(ies)",
            options=categories,
            default=st.session_state.selected_categories,
            help="Select one or more categories to filter locations",
            key="category_filter"
        )
        st.session_state.selected_categories = selected_categories
    
    with col3:
        # Count range filter
        count_range = st.slider(
            "Average Count Range",
            min_value=min_count,
            max_value=max_count,
            value=st.session_state.count_range,
            step=100.0,
            help="Filter locations by average pedestrian count range",
            key="count_range_filter"
        )
        st.session_state.count_range = count_range
    
    # Show filter results
    if len(st.session_state.selected_boroughs) > 0 and len(st.session_state.selected_categories) > 0:
        st.info(f"Showing {len(filtered_df)} of {len(df)} locations")
    else:
        st.info("Select at least one borough and one category to see locations")

# TAB 2: Comparison
with tab2:
    st.header("Comparison Mode")
    
    comparison_enabled = st.checkbox("Enable Comparison Mode", value=False, 
                                     help="Compare two groups side-by-side (e.g., two boroughs or categories)")
    
    if comparison_enabled:
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            st.subheader("Group 1")
            comp_type_1 = st.radio("Compare by:", ["Borough", "Category"], key="comp_type_1", horizontal=True)
            if comp_type_1 == "Borough":
                group1_options = sorted(df['Borough'].unique())
            else:
                group1_options = sorted(df['Category'].unique())
            group1_selection = st.selectbox("Select Group 1:", group1_options, key="group1_select")
        
        with comp_col2:
            st.subheader("Group 2")
            comp_type_2 = st.radio("Compare by:", ["Borough", "Category"], key="comp_type_2", horizontal=True)
            if comp_type_2 == "Borough":
                group2_options = sorted(df['Borough'].unique())
            else:
                group2_options = sorted(df['Category'].unique())
            group2_options = [g for g in group2_options if g != group1_selection]  # Exclude group1
            group2_selection = st.selectbox("Select Group 2:", group2_options, key="group2_select")
        
        if group1_selection and group2_selection:
            # Filter data for each group
            if comp_type_1 == "Borough":
                group1_data = df[df['Borough'] == group1_selection].copy()
            else:
                group1_data = df[df['Category'] == group1_selection].copy()
            
            if comp_type_2 == "Borough":
                group2_data = df[df['Borough'] == group2_selection].copy()
            else:
                group2_data = df[df['Category'] == group2_selection].copy()
            
            if len(group1_data) > 0 and len(group2_data) > 0:
                # Comparison metrics
                st.subheader("Comparison Metrics")
                comp_metrics_col1, comp_metrics_col2, comp_metrics_col3, comp_metrics_col4 = st.columns(4)
                
                with comp_metrics_col1:
                    st.metric(f"{group1_selection} - Count", len(group1_data))
                    st.metric(f"{group2_selection} - Count", len(group2_data))
                    diff_count = len(group1_data) - len(group2_data)
                    st.metric("Difference", diff_count, delta=f"{diff_count/len(group2_data)*100:.1f}%")
                
                with comp_metrics_col2:
                    mean1 = group1_data['avg_recent_count'].mean()
                    mean2 = group2_data['avg_recent_count'].mean()
                    st.metric(f"{group1_selection} - Mean", f"{mean1:,.0f}")
                    st.metric(f"{group2_selection} - Mean", f"{mean2:,.0f}")
                    diff_mean = mean1 - mean2
                    pct_diff = (diff_mean / mean2 * 100) if mean2 > 0 else 0
                    st.metric("Difference", f"{diff_mean:,.0f}", delta=f"{pct_diff:.1f}%")
                
                with comp_metrics_col3:
                    median1 = group1_data['avg_recent_count'].median()
                    median2 = group2_data['avg_recent_count'].median()
                    st.metric(f"{group1_selection} - Median", f"{median1:,.0f}")
                    st.metric(f"{group2_selection} - Median", f"{median2:,.0f}")
                    diff_median = median1 - median2
                    pct_diff_med = (diff_median / median2 * 100) if median2 > 0 else 0
                    st.metric("Difference", f"{diff_median:,.0f}", delta=f"{pct_diff_med:.1f}%")
                
                with comp_metrics_col4:
                    max1 = group1_data['avg_recent_count'].max()
                    max2 = group2_data['avg_recent_count'].max()
                    st.metric(f"{group1_selection} - Max", f"{max1:,.0f}")
                    st.metric(f"{group2_selection} - Max", f"{max2:,.0f}")
                    diff_max = max1 - max2
                    pct_diff_max = (diff_max / max2 * 100) if max2 > 0 else 0
                    st.metric("Difference", f"{diff_max:,.0f}", delta=f"{pct_diff_max:.1f}%")
                
                # Comparison visualization
                st.subheader("Comparison Visualization")
                comp_viz_col1, comp_viz_col2 = st.columns(2)
                
                with comp_viz_col1:
                    # Side-by-side bar chart
                    fig_comp, ax_comp = plt.subplots(figsize=(10, 6))
                    categories = ['Count', 'Mean', 'Median', 'Max']
                    group1_values = [
                        len(group1_data),
                        group1_data['avg_recent_count'].mean(),
                        group1_data['avg_recent_count'].median(),
                        group1_data['avg_recent_count'].max()
                    ]
                    group2_values = [
                        len(group2_data),
                        group2_data['avg_recent_count'].mean(),
                        group2_data['avg_recent_count'].median(),
                        group2_data['avg_recent_count'].max()
                    ]
                    
                    x = np.arange(len(categories))
                    width = 0.35
                    
                    bars1 = ax_comp.bar(x - width/2, group1_values, width, label=group1_selection, color='#3498DB', alpha=0.8)
                    bars2 = ax_comp.bar(x + width/2, group2_values, width, label=group2_selection, color='#E74C3C', alpha=0.8)
                    
                    ax_comp.set_ylabel('Value', fontsize=12, fontweight='bold')
                    ax_comp.set_title(f'Comparison: {group1_selection} vs {group2_selection}', 
                                     fontsize=14, fontweight='bold', pad=15)
                    ax_comp.set_xticks(x)
                    ax_comp.set_xticklabels(categories)
                    ax_comp.legend(fontsize=11)
                    ax_comp.grid(axis='y', alpha=0.3, linestyle='--')
                    ax_comp.spines['top'].set_visible(False)
                    ax_comp.spines['right'].set_visible(False)
                    
                    # Add value labels on bars
                    for bars in [bars1, bars2]:
                        for bar in bars:
                            height = bar.get_height()
                            if height > 0:
                                label = f'{height:,.0f}' if height >= 1 else f'{height:.1f}'
                                ax_comp.text(bar.get_x() + bar.get_width()/2., height,
                                            label, ha='center', va='bottom', fontsize=9)
                    
                    plt.tight_layout()
                    st.pyplot(fig_comp)
                    plt.close(fig_comp)
                
                with comp_viz_col2:
                    # Distribution comparison
                    fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
                    
                    ax_dist.hist(group1_data['avg_recent_count'], bins=20, alpha=0.6, 
                               label=group1_selection, color='#3498DB', edgecolor='black')
                    ax_dist.hist(group2_data['avg_recent_count'], bins=20, alpha=0.6, 
                               label=group2_selection, color='#E74C3C', edgecolor='black')
                    
                    ax_dist.set_xlabel('Average Count', fontsize=12, fontweight='bold')
                    ax_dist.set_ylabel('Frequency', fontsize=12, fontweight='bold')
                    ax_dist.set_title(f'Distribution Comparison', fontsize=14, fontweight='bold', pad=15)
                    ax_dist.legend(fontsize=11)
                    ax_dist.grid(axis='y', alpha=0.3, linestyle='--')
                    ax_dist.spines['top'].set_visible(False)
                    ax_dist.spines['right'].set_visible(False)
                    
                    plt.tight_layout()
                    st.pyplot(fig_dist)
                    plt.close(fig_dist)
                
                # Detailed comparison table
                st.subheader("Detailed Comparison")
                comp_table_data = {
                    'Metric': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', '25th Percentile', '75th Percentile'],
                    group1_selection: [
                        len(group1_data),
                        f"{group1_data['avg_recent_count'].mean():,.1f}",
                        f"{group1_data['avg_recent_count'].median():,.1f}",
                        f"{group1_data['avg_recent_count'].std():,.1f}",
                        f"{group1_data['avg_recent_count'].min():,.1f}",
                        f"{group1_data['avg_recent_count'].max():,.1f}",
                        f"{group1_data['avg_recent_count'].quantile(0.25):,.1f}",
                        f"{group1_data['avg_recent_count'].quantile(0.75):,.1f}"
                    ],
                    group2_selection: [
                        len(group2_data),
                        f"{group2_data['avg_recent_count'].mean():,.1f}",
                        f"{group2_data['avg_recent_count'].median():,.1f}",
                        f"{group2_data['avg_recent_count'].std():,.1f}",
                        f"{group2_data['avg_recent_count'].min():,.1f}",
                        f"{group2_data['avg_recent_count'].max():,.1f}",
                        f"{group2_data['avg_recent_count'].quantile(0.25):,.1f}",
                        f"{group2_data['avg_recent_count'].quantile(0.75):,.1f}"
                    ]
                }
                comp_table_df = pd.DataFrame(comp_table_data)
                
                # Calculate differences
                comp_table_df['Difference'] = comp_table_df.apply(
                    lambda row: f"{float(row[group1_selection].replace(',', '')) - float(row[group2_selection].replace(',', '')):,.1f}" 
                    if row['Metric'] != 'Count' else str(int(row[group1_selection]) - int(row[group2_selection])),
                    axis=1
                )
                
                comp_table_df['% Difference'] = comp_table_df.apply(
                    lambda row: f"{((float(row[group1_selection].replace(',', '')) - float(row[group2_selection].replace(',', ''))) / float(row[group2_selection].replace(',', '')) * 100):.1f}%" 
                    if row['Metric'] != 'Count' and float(row[group2_selection].replace(',', '')) > 0 else "N/A",
                    axis=1
                )
                
                st.dataframe(comp_table_df, use_container_width=True, hide_index=True)
            else:
                st.warning("One or both selected groups have no data. Please select different groups.")
        else:
            st.info("Please select both groups to compare.")
    else:
        st.info("Enable comparison mode to compare groups.")

# TAB 3: Statistics
with tab3:
    st.header("Summary Statistics & Visualizations")
    
    if len(filtered_df) > 0:
        # Key Metrics Row
        st.subheader("Key Metrics")
        metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    
    with metric_col1:
        st.metric("Total Locations", len(filtered_df))
    with metric_col2:
        st.metric("Mean Count", f"{filtered_df['avg_recent_count'].mean():,.0f}")
    with metric_col3:
        st.metric("Median Count", f"{filtered_df['avg_recent_count'].median():,.0f}")
    with metric_col4:
        st.metric("Max Count", f"{filtered_df['avg_recent_count'].max():,.0f}")
    with metric_col5:
        st.metric("Min Count", f"{filtered_df['avg_recent_count'].min():,.0f}")
    
    # Summary Statistics Section with better styling
    st.subheader("Summary Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Summary by Category")
        if 'Category' in filtered_df.columns:
            cat_summary = filtered_df.groupby('Category')['avg_recent_count'].agg([
                'count', 'mean', 'median', 'std', 'min', 'max'
            ]).round(1).sort_values('mean', ascending=False)
            # Rename columns for better display
            cat_summary.columns = ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max']
            st.dataframe(
                cat_summary.style.format({
                    'Count': '{:.0f}',
                    'Mean': '{:,.0f}',
                    'Median': '{:,.0f}',
                    'Std Dev': '{:,.0f}',
                    'Min': '{:,.0f}',
                    'Max': '{:,.0f}'
                }),
                use_container_width=True,
                height=200
            )
        else:
            st.info("Category data not available for filtered results")
    
    with col2:
        st.markdown("##### Summary by Borough")
        if 'Borough' in filtered_df.columns:
            borough_summary = filtered_df.groupby('Borough')['avg_recent_count'].agg([
                'count', 'mean', 'median', 'std', 'min', 'max'
            ]).round(1).sort_values('mean', ascending=False)
            # Rename columns for better display
            borough_summary.columns = ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max']
            st.dataframe(
                borough_summary.style.format({
                    'Count': '{:.0f}',
                    'Mean': '{:,.0f}',
                    'Median': '{:,.0f}',
                    'Std Dev': '{:,.0f}',
                    'Min': '{:,.0f}',
                    'Max': '{:,.0f}'
                }),
                use_container_width=True,
                height=200
            )
        else:
            st.info("Borough data not available for filtered results")
    
    # Visualizations with improved styling
    st.subheader("Visualizations")
    
    # Set improved style
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16
    
    # Chart 1: Average Count by Category (Bar Chart)
    if 'Category' in filtered_df.columns and len(filtered_df['Category'].unique()) > 0:
        fig1, ax1 = plt.subplots(figsize=(11, 6))
        cat_order = ['Global', 'Regional', 'Neighborhood', 'Community', 'Baseline']
        cat_order = [c for c in cat_order if c in filtered_df['Category'].unique()]
        cat_means = filtered_df.groupby('Category')['avg_recent_count'].mean().reindex(cat_order, fill_value=0)
        
        # Color mapping matching map colors
        color_map = {
            'Global': '#FF0000',
            'Regional': '#FF8C00',
            'Neighborhood': '#0066CC',
            'Community': '#00AA00',
            'Baseline': '#808080'
        }
        colors = [color_map.get(cat, '#808080') for cat in cat_means.index]
        
        bars = ax1.bar(cat_means.index, cat_means.values, color=colors, edgecolor='white', linewidth=1.5, alpha=0.85)
        ax1.set_ylabel('Mean Average Count', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Category', fontsize=12, fontweight='bold')
        ax1.set_title('Mean Pedestrian Counts by Category', fontsize=15, fontweight='bold', pad=15)
        ax1.tick_params(axis='x', rotation=0)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:,.0f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)
    
    # Chart 2: Distribution by Category (Boxplot)
    if 'Category' in filtered_df.columns and len(filtered_df['Category'].unique()) > 0:
        fig2, ax2 = plt.subplots(figsize=(11, 6))
        cat_order = ['Global', 'Regional', 'Neighborhood', 'Community', 'Baseline']
        cat_order = [c for c in cat_order if c in filtered_df['Category'].unique()]
        
        data_for_box = [filtered_df[filtered_df['Category'] == cat]['avg_recent_count'].values 
                       for cat in cat_order if cat in filtered_df['Category'].unique()]
        
        if data_for_box:
            bp = ax2.boxplot(data_for_box, labels=[c for c in cat_order if c in filtered_df['Category'].unique()],
                           patch_artist=True, widths=0.6)
            
            # Color the boxes
            color_map = {
                'Global': '#FF0000',
                'Regional': '#FF8C00',
                'Neighborhood': '#0066CC',
                'Community': '#00AA00',
                'Baseline': '#808080'
            }
            colors = [color_map.get(cat, '#808080') for cat in cat_order if cat in filtered_df['Category'].unique()]
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                patch.set_edgecolor('white')
                patch.set_linewidth(1.5)
            
            # Style the other elements
            for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
                if element in bp:
                    plt.setp(bp[element], color='#333333', linewidth=1.5)
            
            ax2.set_ylabel('Average Count', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Category', fontsize=12, fontweight='bold')
            ax2.set_title('Distribution of Pedestrian Counts by Category', fontsize=15, fontweight='bold', pad=15)
            ax2.tick_params(axis='x', rotation=0)
            ax2.set_yscale('symlog')
            ax2.grid(axis='y', alpha=0.3, linestyle='--')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
    
    # Chart 3: Borough vs Category Heatmap
    if 'Borough' in filtered_df.columns and 'Category' in filtered_df.columns:
        if len(filtered_df['Borough'].unique()) > 1 and len(filtered_df['Category'].unique()) > 1:
            fig3, ax3 = plt.subplots(figsize=(11, 6))
            pivot = filtered_df.pivot_table(
                index='Borough', 
                columns='Category', 
                values='avg_recent_count', 
                aggfunc='mean'
            ).fillna(0)
            
            # Reorder columns and rows for better visualization
            cat_order = ['Global', 'Regional', 'Neighborhood', 'Community', 'Baseline']
            cat_order = [c for c in cat_order if c in pivot.columns]
            pivot = pivot[cat_order] if cat_order else pivot
            
            sns.heatmap(pivot, annot=True, fmt=".0f", cmap='Blues', 
                       cbar_kws={'label': 'Average Count', 'shrink': 0.8}, 
                       ax=ax3, linewidths=0.5, linecolor='white',
                       annot_kws={'size': 10, 'weight': 'bold'})
            ax3.set_title('Average Pedestrian Count by Borough and Category', 
                         fontsize=15, fontweight='bold', pad=15)
            ax3.set_xlabel('Category', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Borough', fontsize=12, fontweight='bold')
            ax3.tick_params(axis='both', labelsize=10)
            
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)
    
    # Chart 4: Average Count by Borough (Bar Chart)
    if 'Borough' in filtered_df.columns and len(filtered_df['Borough'].unique()) > 0:
        fig4, ax4 = plt.subplots(figsize=(11, 6))
        borough_means = filtered_df.groupby('Borough')['avg_recent_count'].mean().sort_values(ascending=False)
        
        bars = ax4.bar(borough_means.index, borough_means.values, 
                      color='#3498DB', edgecolor='white', linewidth=1.5, alpha=0.85)
        ax4.set_ylabel('Mean Average Count', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Borough', fontsize=12, fontweight='bold')
        ax4.set_title('Mean Pedestrian Counts by Borough', fontsize=15, fontweight='bold', pad=15)
        ax4.tick_params(axis='x', rotation=0)
        ax4.grid(axis='y', alpha=0.3, linestyle='--')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:,.0f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)
    
    else:
        st.info("Select filters above to see summary statistics and visualizations for the filtered data.")

# TAB 4: Time Series
with tab4:
    st.header("Time Series Analysis")
    
    if df_raw is not None and time_df is not None and len(time_cols) > 0:
        st.info(f"Historical data available: {len(time_cols)} time periods from {time_df['date'].min().strftime('%B %Y')} to {time_df['date'].max().strftime('%B %Y')}")
        
        # Filter time series data to match current filters
        if len(filtered_df) > 0:
            filtered_objectids = filtered_df['OBJECTID'].astype(str).tolist()
            ts_data = df_raw[df_raw['OBJECTID'].isin(filtered_objectids)].copy()
            
            if len(ts_data) > 0:
                # Allow user to select specific locations for time series
                location_options = ts_data.apply(
                    lambda x: f"Loc {x['Loc']}: {x['Street_Nam']} ({x['Borough']})", axis=1
                ).tolist()
                
                col_ts1, col_ts2 = st.columns([2, 1])
                with col_ts1:
                    selected_locations = st.multiselect(
                        "Select locations for time series (leave empty for all filtered locations)",
                        options=location_options,
                        default=[],
                        key="ts_location_select"
                    )
                
                with col_ts2:
                    period_filter = st.multiselect(
                        "Select time periods",
                        options=['AM', 'PM', 'MD'],
                        default=['AM', 'PM', 'MD'],
                        key="ts_period_filter"
                    )
                
                # Filter data based on selections
                if len(selected_locations) > 0:
                    selected_locs = [int(loc.split(':')[0].replace('Loc ', '')) for loc in selected_locations]
                    ts_data = ts_data[ts_data['Loc'].astype(str).isin([str(l) for l in selected_locs])]
                
                # Filter time columns by period
                if len(period_filter) > 0:
                    filtered_time_cols = [col for col in time_cols if any(p in col for p in period_filter)]
                else:
                    filtered_time_cols = time_cols
                
                if len(ts_data) > 0 and len(filtered_time_cols) > 0:
                    # Convert time columns to numeric
                    for col in filtered_time_cols:
                        ts_data[col] = pd.to_numeric(ts_data[col].str.replace(',', ''), errors='coerce')
                    
                    # Calculate time series statistics
                    ts_stats = []
                    for idx, row in ts_data.iterrows():
                        values = row[filtered_time_cols].values
                        valid_values = values[~pd.isna(values)]
                        if len(valid_values) > 0:
                            ts_stats.append({
                                'OBJECTID': row['OBJECTID'],
                                'Loc': row['Loc'],
                                'Street': row['Street_Nam'],
                                'Borough': row['Borough'],
                                'Mean': valid_values.mean(),
                                'Std': valid_values.std(),
                                'Min': valid_values.min(),
                                'Max': valid_values.max(),
                                'Data Points': len(valid_values)
                            })
                    
                    if len(ts_stats) > 0:
                        ts_stats_df = pd.DataFrame(ts_stats)
                        
                        # Time series visualization
                        st.subheader("Time Series Trends")
                        
                        # Aggregate data for plotting
                        time_series_agg = []
                        for col in filtered_time_cols:
                            col_data = ts_data[col].values
                            valid_data = col_data[~pd.isna(col_data)]
                            if len(valid_data) > 0:
                                # Find corresponding date
                                col_info = time_df[time_df['column'] == col].iloc[0]
                                time_series_agg.append({
                                    'date': col_info['date'],
                                    'period': col_info['period'],
                                    'mean': valid_data.mean(),
                                    'median': np.median(valid_data),
                                    'count': len(valid_data)
                                })
                        
                        if len(time_series_agg) > 0:
                            ts_plot_df = pd.DataFrame(time_series_agg).sort_values('date')
                            
                            # Create time series plot
                            fig_ts, ax_ts = plt.subplots(figsize=(14, 6))
                            
                            # Plot by period
                            for period in ['AM', 'PM', 'MD']:
                                period_data = ts_plot_df[ts_plot_df['period'] == period]
                                if len(period_data) > 0:
                                    ax_ts.plot(period_data['date'], period_data['mean'], 
                                              marker='o', label=f'{period} Period', linewidth=2, markersize=6)
                            
                            ax_ts.set_xlabel('Date', fontsize=12, fontweight='bold')
                            ax_ts.set_ylabel('Mean Pedestrian Count', fontsize=12, fontweight='bold')
                            ax_ts.set_title('Time Series: Mean Pedestrian Counts Over Time', 
                                           fontsize=15, fontweight='bold', pad=15)
                            ax_ts.legend(fontsize=11)
                            ax_ts.grid(axis='y', alpha=0.3, linestyle='--')
                            ax_ts.spines['top'].set_visible(False)
                            ax_ts.spines['right'].set_visible(False)
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            st.pyplot(fig_ts)
                            plt.close(fig_ts)
                            
                            # Show statistics table
                            st.subheader("Time Series Statistics by Location")
                            st.dataframe(ts_stats_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("No valid time series data available for selected filters.")
                    else:
                        st.info("No time series statistics could be calculated.")
                else:
                    st.info("No data available for selected filters.")
            else:
                st.info("No matching locations found in historical data.")
        else:
            st.info("Please apply filters above to see time series analysis for filtered locations.")
    else:
        st.info("Time series data not available. Historical count data may not be loaded.")
        st.caption("Note: Time series analysis requires access to the raw Pedestrian_Counts.csv file with historical measurements.")

# TAB 5: Top Sites
with tab5:
    st.header("Top Pedestrian Count Sites")
    
    # Get filtered data for top sites (use the same filter logic as Filters section)
    # If filters are applied, use filtered_df; otherwise use full df
    if len(st.session_state.selected_boroughs) > 0 and len(st.session_state.selected_categories) > 0:
        top_sites_df = df[
            (df['Borough'].isin(st.session_state.selected_boroughs)) &
            (df['Category'].isin(st.session_state.selected_categories)) &
            (df['avg_recent_count'] >= st.session_state.count_range[0]) &
            (df['avg_recent_count'] <= st.session_state.count_range[1])
        ]
    else:
        # If no filters selected, show top sites from full dataset
        top_sites_df = df.copy()
    
    if len(top_sites_df) > 0:
        # Allow user to select number of top sites
        col1, col2 = st.columns([1, 3])
        with col1:
            top_n = st.selectbox(
                "Number of top sites to display:",
                options=[10, 20, 30, 50, 100],
                index=0,
                key="top_n_sites"
            )
    
        # Sort by avg_recent_count descending
        top_sites = top_sites_df.nlargest(top_n, 'avg_recent_count').copy()
        
        # Select and format columns for display
        display_cols = ['Loc', 'Street_Nam_clean', 'Borough', 'Category', 'avg_recent_count']
        # Only include columns that exist
        display_cols = [col for col in display_cols if col in top_sites.columns]
        
        top_sites_display = top_sites[display_cols].copy()
        
        # Rename columns for better display
        column_mapping = {
            'Loc': 'Location #',
            'Street_Nam_clean': 'Street Name',
            'Borough': 'Borough',
            'Category': 'Category',
            'avg_recent_count': 'Avg Count'
        }
        top_sites_display = top_sites_display.rename(columns=column_mapping)
        
        # Format the average count column
        if 'Avg Count' in top_sites_display.columns:
            top_sites_display['Avg Count'] = top_sites_display['Avg Count'].apply(
                lambda x: f"{x:,.0f}" if pd.notnull(x) else "N/A"
            )
        
        # Add rank column
        top_sites_display.insert(0, 'Rank', range(1, len(top_sites_display) + 1))
        
        # Display the table
        st.markdown(f"**Top {top_n} sites by average pedestrian count:**")
        st.dataframe(
            top_sites_display,
            use_container_width=True,
            hide_index=True,
            height=400
        )
        
        # Add download button for top sites
        csv = top_sites_display.to_csv(index=False)
        st.download_button(
            label="Download Top Sites as CSV",
            data=csv,
            file_name=f"top_{top_n}_sites.csv",
            mime="text/csv",
            key="download_top_sites"
        )
        
        # Show some statistics about top sites
        st.markdown("### Top Sites Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Highest Count", f"{top_sites['avg_recent_count'].max():,.0f}")
        with col2:
            st.metric("Lowest in Top N", f"{top_sites['avg_recent_count'].min():,.0f}")
        with col3:
            st.metric("Average of Top N", f"{top_sites['avg_recent_count'].mean():,.0f}")
        with col4:
            st.metric("Median of Top N", f"{top_sites['avg_recent_count'].median():,.0f}")
        
        # Show category distribution in top sites
        if 'Category' in top_sites.columns:
            st.markdown("### Category Distribution in Top Sites")
            cat_counts = top_sites['Category'].value_counts()
            cat_df = pd.DataFrame({
                'Category': cat_counts.index,
                'Count': cat_counts.values,
                'Percentage': (cat_counts.values / len(top_sites) * 100).round(1)
            })
            st.dataframe(cat_df, use_container_width=True, hide_index=True)
        
        # Show borough distribution in top sites
        if 'Borough' in top_sites.columns:
            st.markdown("### Borough Distribution in Top Sites")
            borough_counts = top_sites['Borough'].value_counts()
            borough_df = pd.DataFrame({
                'Borough': borough_counts.index,
                'Count': borough_counts.values,
                'Percentage': (borough_counts.values / len(top_sites) * 100).round(1)
            })
            st.dataframe(borough_df, use_container_width=True, hide_index=True)
        
    else:
        st.info("No data available. Please check your filters or data files.")

# TAB 6: Export
with tab6:
    st.header("Export Data")
    
    if len(filtered_df) > 0:
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            # Export filtered data as CSV
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Filtered Data (CSV)",
                data=csv_data,
                file_name=f"pedestrian_counts_filtered_{len(filtered_df)}_locations.csv",
                mime="text/csv",
                key="download_filtered_csv"
            )
        
        with export_col2:
            # Export summary statistics
            if 'Category' in filtered_df.columns and 'Borough' in filtered_df.columns:
                summary_data = {
                    'Summary Type': [],
                    'Group': [],
                    'Count': [],
                    'Mean': [],
                    'Median': [],
                    'Std Dev': [],
                    'Min': [],
                    'Max': []
                }
                
                # Category summary
                cat_summary = filtered_df.groupby('Category')['avg_recent_count'].agg([
                    'count', 'mean', 'median', 'std', 'min', 'max'
                ]).round(1)
                for cat, row in cat_summary.iterrows():
                    summary_data['Summary Type'].append('Category')
                    summary_data['Group'].append(cat)
                    summary_data['Count'].append(int(row['count']))
                    summary_data['Mean'].append(round(row['mean'], 1))
                    summary_data['Median'].append(round(row['median'], 1))
                    summary_data['Std Dev'].append(round(row['std'], 1))
                    summary_data['Min'].append(round(row['min'], 1))
                    summary_data['Max'].append(round(row['max'], 1))
                
                # Borough summary
                borough_summary = filtered_df.groupby('Borough')['avg_recent_count'].agg([
                    'count', 'mean', 'median', 'std', 'min', 'max'
                ]).round(1)
                for borough, row in borough_summary.iterrows():
                    summary_data['Summary Type'].append('Borough')
                    summary_data['Group'].append(borough)
                    summary_data['Count'].append(int(row['count']))
                    summary_data['Mean'].append(round(row['mean'], 1))
                    summary_data['Median'].append(round(row['median'], 1))
                    summary_data['Std Dev'].append(round(row['std'], 1))
                    summary_data['Min'].append(round(row['min'], 1))
                    summary_data['Max'].append(round(row['max'], 1))
                
                summary_df = pd.DataFrame(summary_data)
                summary_csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="Download Summary Statistics (CSV)",
                    data=summary_csv,
                    file_name="pedestrian_counts_summary_statistics.csv",
                    mime="text/csv",
                    key="download_summary_csv"
                )
            else:
                st.info("Summary export not available")
        
        with export_col3:
            # Export current filter settings
            filter_settings = {
                'Filter': ['Selected Boroughs', 'Selected Categories', 'Count Range Min', 'Count Range Max', 'Total Locations'],
                'Value': [
                    ', '.join(st.session_state.selected_boroughs),
                    ', '.join(st.session_state.selected_categories),
                    str(st.session_state.count_range[0]),
                    str(st.session_state.count_range[1]),
                    str(len(filtered_df))
                ]
            }
            filter_df = pd.DataFrame(filter_settings)
            filter_csv = filter_df.to_csv(index=False)
            st.download_button(
                label="Download Filter Settings (CSV)",
                data=filter_csv,
                file_name="pedestrian_counts_filter_settings.csv",
                mime="text/csv",
                key="download_filter_settings"
            )
    else:
        st.info("No filtered data available. Please apply filters to export data.")

