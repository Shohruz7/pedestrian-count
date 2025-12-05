# NYC Pedestrian Count Dashboard

Interactive web dashboard for visualizing pedestrian count data from the [New York City Department of Transportation (NYC DOT)](https://www.nyc.gov/html/dot/html/home/home.shtml).

Data acquired from [NYC Open Data](https://opendata.cityofnewyork.us/).

## Features

- 🗺️ **Interactive Maps**: View pedestrian counts on an interactive map with marker clustering and heatmap visualization
- 📊 **Statistics & Visualizations**: Comprehensive statistics with bar charts for borough and category comparisons
- 🔍 **Advanced Filtering**: Filter by borough, category, count range, and search
- 📈 **Time Series Analysis**: Analyze pedestrian count trends over time by location
- 🔄 **Comparison Mode**: Compare statistics between different boroughs or categories with visual charts
- 💾 **Export Functionality**: Export filtered data as CSV or GeoJSON
- 🌓 **Dark/Light Mode**: Toggle between light and dark themes
- ⌨️ **Keyboard Shortcuts**: Navigate quickly with keyboard shortcuts
- 📱 **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices

## Tech Stack

### Frontend
- **React 18** with Vite
- **Material-UI (MUI)** for UI components
- **Leaflet & React-Leaflet** for interactive maps
- **Recharts** for data visualizations
- **CSV/GeoJSON** data files for data loading

## Prerequisites

- **Node.js 18+** and npm
- **Git**

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd pedestrian-count
```

### 2. Frontend Setup

#### Install Dependencies

```bash
cd frontend
npm install
```

#### Run Development Server

```bash
npm run dev
```

The app will be available at `http://localhost:3000`

#### Build for Production

```bash
npm run build
```

The optimized build will be in the `dist` directory.

## Project Structure

```
pedestrian-count/
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── services/       # CSV data services
│   │   ├── hooks/         # Custom React hooks
│   │   └── App.jsx        # Main app component
│   ├── public/            # Static assets and data files
│   │   ├── pedestrian_data.csv
│   │   └── pedestrian_data.geojson
│   └── package.json       # Node dependencies
├── data_clean/            # Processed data files
├── data-raw/              # Raw data files
└── notebooks/             # Jupyter notebooks for data analysis
```

## Data Files

The application uses CSV and GeoJSON files located in `frontend/public/`:
- `pedestrian_data.csv` - Main pedestrian count data
- `pedestrian_data.geojson` - Geospatial data for map visualization

These files are loaded directly by the frontend, no backend required.

## Keyboard Shortcuts

- `1-6`: Switch between tabs
- `Ctrl+R` / `Cmd+R`: Refresh data
- `Ctrl+/` / `Cmd+/`: Show keyboard shortcuts help

## Data Sources

Data is sourced from [NYC Open Data](https://opendata.cityofnewyork.us/):
- [Pedestrian Mobility Plan Pedestrian Demand Map](https://data.cityofnewyork.us/Transportation/Pedestrian-Mobility-Plan-Pedestrian-Demand-Map/c4kr-96ik)
- [Bi-Annual Pedestrian Counts](https://data.cityofnewyork.us/Transportation/Bi-Annual-Pedestrian-Counts/2de2-6x2h)

## Development

### Data Analysis

Jupyter notebooks in the `notebooks/` directory contain data exploration and analysis:
- `01_exploration.ipynb` - Initial data exploration
- `02_clean_merge.ipynb` - Data cleaning and merging
- `03_visualization.ipynb` - Visualization analysis

### Updating Data Files

To update the data files:
1. Process your data using the notebooks
2. Copy the cleaned CSV and GeoJSON files to `frontend/public/`
3. Restart the development server

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

See LICENSE file for details.

## Support

For issues or questions, please open an issue on the repository.
