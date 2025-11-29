# NYC Pedestrian Count Dashboard

Interactive web dashboard for visualizing pedestrian count data from the [New York City Department of Transportation (NYC DOT)](https://www.nyc.gov/html/dot/html/home/home.shtml).

Data acquired from [NYC Open Data](https://opendata.cityofnewyork.us/).

## Features

- 🗺️ **Interactive Maps**: View pedestrian counts on an interactive map with marker clustering and heatmap visualization
- 📊 **Statistics & Visualizations**: Comprehensive statistics with bar charts for borough and category comparisons
- 🔍 **Advanced Filtering**: Filter by borough, category, count range, date range, and search
- 📈 **Time Series Analysis**: Analyze pedestrian count trends over time by location
- 🔄 **Comparison Mode**: Compare statistics between different boroughs or categories
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
- **Axios** for API requests

### Backend
- **Flask** REST API
- **PostgreSQL** with **PostGIS** for geospatial data
- **SQLAlchemy** ORM
- **Flask-Limiter** for rate limiting
- **Flask-CORS** for cross-origin requests

## Prerequisites

- **Python 3.9+**
- **Node.js 18+** and npm
- **PostgreSQL 14+** with PostGIS extension
- **Git**

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd pedestrian-count
```

### 2. Backend Setup

#### Install Python Dependencies

```bash
# Create virtual environment (if using venv)
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
cd backend
pip install -r requirements.txt
```

#### Set Up Database

1. **Install PostgreSQL** and ensure PostGIS extension is available

2. **Create database**:
```bash
createdb pedestrian_count_db
```

3. **Enable PostGIS**:
```bash
psql -d pedestrian_count_db -c "CREATE EXTENSION IF NOT EXISTS postgis;"
```

4. **Run schema**:
```bash
psql -d pedestrian_count_db -f database/schema.sql
```

5. **Load data** (optional - for testing with database):
```bash
python migrate_data.py
```

#### Configure Environment Variables

Create a `.env` file in the `backend` directory:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=pedestrian_count_db
DB_USER=postgres
DB_PASSWORD=your_password
SECRET_KEY=your-secret-key-here
CORS_ORIGINS=http://localhost:3000

# NYC Open Data API (optional)
NYC_API_DEMAND_MAP_ENDPOINT=https://data.cityofnewyork.us/resource/fwpa-qxaf.json
NYC_API_COUNT_ENDPOINT=https://data.cityofnewyork.us/resource/cqsj-cfgu.json
USE_NYC_API=false
```

#### Run Backend Server

```bash
# From backend directory
python run.py
```

The API will be available at `http://localhost:5000`

### 3. Frontend Setup

#### Install Dependencies

```bash
cd frontend
npm install
```

#### Configure Environment

Create a `.env` file in the `frontend` directory (or copy from `.env.example`):

```env
VITE_API_BASE_URL=http://localhost:5000/api
```

**Note**: The frontend is configured to connect to the Flask backend API. Make sure the backend server is running before starting the frontend. The default API URL is `http://localhost:5000/api`. If your backend runs on a different port or host, update the `VITE_API_BASE_URL` accordingly.

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
├── backend/                 # Flask backend
│   ├── app/
│   │   ├── models/         # SQLAlchemy models
│   │   ├── routes/         # API route handlers
│   │   ├── services/       # Business logic
│   │   └── config.py      # Configuration
│   ├── database/
│   │   └── schema.sql      # Database schema
│   ├── migrate_data.py     # Data migration script
│   └── requirements.txt    # Python dependencies
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── services/       # API services
│   │   ├── hooks/         # Custom React hooks
│   │   └── App.jsx        # Main app component
│   ├── public/            # Static assets
│   └── package.json       # Node dependencies
├── data_clean/            # Processed data files
├── data-raw/              # Raw data files
└── notebooks/             # Jupyter notebooks for data analysis
```

## API Endpoints

### Locations
- `GET /api/locations` - Get all locations with filters (add `?source=nyc` to use NYC API)
- `GET /api/locations/nyc` - Get locations directly from NYC Open Data API
- `GET /api/locations/<id>` - Get single location
- `GET /api/locations/bounds` - Get locations within map bounds

### Statistics
- `GET /api/statistics/summary` - Overall summary statistics
- `GET /api/statistics/by-borough` - Statistics by borough
- `GET /api/statistics/by-category` - Statistics by category
- `GET /api/statistics/top-sites` - Top N sites by count

### Time Series
- `GET /api/counts/time-series/<location_id>` - Time series for a location
- `GET /api/counts/time-series/aggregate` - Aggregated time series

### Comparison
- `POST /api/comparison` - Compare two groups

### Export
- `GET /api/export/csv` - Export as CSV
- `GET /api/export/geojson` - Export as GeoJSON

### Health
- `GET /api/health` - Health check

## Keyboard Shortcuts

- `1-6`: Switch between tabs
- `Ctrl+R` / `Cmd+R`: Refresh data
- `Ctrl+/` / `Cmd+/`: Show keyboard shortcuts help

## Data Sources

The application supports two data sources:

### Database (Default)
- Local PostgreSQL database with pre-loaded data
- Faster queries and advanced filtering
- Supports all features including time series analysis

### NYC Open Data API (Optional)
- Live data from NYC Open Data endpoints
- No database setup required
- Automatically fetches latest data

**NYC Open Data Endpoints:**
- [Pedestrian Mobility Plan Pedestrian Demand Map](https://data.cityofnewyork.us/Transportation/Pedestrian-Mobility-Plan-Pedestrian-Demand-Map/c4kr-96ik?referrer=embed) - `https://data.cityofnewyork.us/resource/fwpa-qxaf.json`
- [Bi-Annual Pedestrian Counts](https://data.cityofnewyork.us/Transportation/Bi-Annual-Pedestrian-Counts/2de2-6x2h) - `https://data.cityofnewyork.us/resource/cqsj-cfgu.json`

See [NYC_API_INTEGRATION.md](NYC_API_INTEGRATION.md) for details on using the NYC API.

## Development Tools

- [uv](https://docs.astral.sh/uv/) - Package and project management
- [Pandas](https://pandas.pydata.org/) - Data analysis and manipulation
- [GeoPandas](https://geopandas.org/) - Working with geospatial data

## Rate Limiting

The API includes rate limiting to prevent abuse:
- Default: 200 requests per day, 50 per hour
- Location endpoints: 30 requests per minute
- Statistics endpoints: 60 requests per minute
- Export endpoints: 20 requests per minute

## Logging

Backend requests and responses are logged to:
- Console output
- `backend/app.log` file

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

See LICENSE file for details.

## Support

For issues or questions, please open an issue on the repository.
