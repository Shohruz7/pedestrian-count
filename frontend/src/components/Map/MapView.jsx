import { useEffect, useState, useRef } from 'react'
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet'
import { useTheme } from '@mui/material'
import L from 'leaflet'
import 'leaflet.markercluster'
import 'leaflet.markercluster/dist/MarkerCluster.css'
import 'leaflet.markercluster/dist/MarkerCluster.Default.css'
import { getLocations } from '../../services/csvDataService'
import { Box, Typography, Alert } from '@mui/material'
import { MapSkeleton } from '../common/LoadingSkeleton'

// Fix for default marker icons in React-Leaflet
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
})

// NYC center coordinates
const NYC_CENTER = [40.7128, -73.9352]
const DEFAULT_ZOOM = 11

function MapBoundsUpdater({ bounds, onBoundsChange }) {
  const map = useMap()
  
  useEffect(() => {
    const handleMoveEnd = () => {
      const newBounds = map.getBounds()
      onBoundsChange([
        newBounds.getSouth(),
        newBounds.getWest(),
        newBounds.getNorth(),
        newBounds.getEast(),
      ].join(','))
    }
    
    map.on('moveend', handleMoveEnd)
    return () => {
      map.off('moveend', handleMoveEnd)
    }
  }, [map, onBoundsChange])
  
  return null
}

function MarkerClusterGroupComponent({ locations }) {
  const map = useMap()
  const clusterGroupRef = useRef(null)

  useEffect(() => {
    if (!clusterGroupRef.current) {
      clusterGroupRef.current = L.markerClusterGroup({
        maxClusterRadius: 50,
      })
      map.addLayer(clusterGroupRef.current)
    }

    // Clear existing markers
    clusterGroupRef.current.clearLayers()

    // Add markers
    locations.forEach((feature) => {
      const { geometry, properties } = feature
      
      if (!geometry || !geometry.coordinates) return
      
      const [lng, lat] = geometry.coordinates
      const count = properties?.avg_recent_count || 0
      const street = properties?.street_name_clean || properties?.street_clean || 'Unknown'
      const borough = properties?.borough || 'Unknown'
      const category = properties?.category || 'Unknown'
      
      const marker = L.marker([lat, lng])
      marker.bindPopup(`
        <div>
          <strong>${street}</strong><br />
          Borough: ${borough}<br />
          Category: ${category}<br />
          Avg Count: ${Math.round(count).toLocaleString()}
        </div>
      `)
      clusterGroupRef.current.addLayer(marker)
    })

    return () => {
      if (clusterGroupRef.current) {
        map.removeLayer(clusterGroupRef.current)
        clusterGroupRef.current = null
      }
    }
  }, [locations, map])

  return null
}

function MapView({ filters }) {
  const theme = useTheme()
  const [locations, setLocations] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [mapBounds, setMapBounds] = useState(null)
  
  // Get tile layer URL based on theme
  const tileUrl = theme.palette.mode === 'dark' 
    ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
    : 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'
  
  const tileAttribution = theme.palette.mode === 'dark'
    ? '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
    : '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'

  useEffect(() => {
    const fetchLocations = async () => {
      setLoading(true)
      setError(null)
      
      try {
        // Build filters object for CSV service
        const filterParams = {}
        
        // Add filters
        if (filters.boroughs?.length > 0) {
          filterParams.boroughs = filters.boroughs
        }
        if (filters.categories?.length > 0) {
          filterParams.categories = filters.categories
        }
        if (filters.minCount !== null && filters.minCount !== '') {
          filterParams.minCount = filters.minCount
        }
        if (filters.maxCount !== null && filters.maxCount !== '') {
          filterParams.maxCount = filters.maxCount
        }
        if (filters.search) {
          filterParams.search = filters.search
        }
        
        const data = await getLocations(filterParams)
        
        if (data.features) {
          setLocations(data.features)
        } else if (data.locations) {
          // Fallback if API returns JSON instead of GeoJSON
          setLocations(data.locations)
        }
      } catch (err) {
        console.error('Error fetching locations:', err)
        setError(err.message || 'Failed to load locations')
      } finally {
        setLoading(false)
      }
    }

    fetchLocations()
  }, [filters])

  const handleBoundsChange = (bounds) => {
    setMapBounds(bounds)
    // Optionally fetch locations within bounds for performance
  }

  if (loading) {
    return <MapSkeleton />
  }

  if (error) {
    return (
      <Box sx={{ p: 2, height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Alert severity="error" sx={{ maxWidth: 500 }}>
          <Typography variant="h6" gutterBottom>Failed to load map</Typography>
          <Typography variant="body2">{error}</Typography>
          <Typography variant="body2" sx={{ mt: 1 }}>
            Please check your connection and try refreshing the page.
          </Typography>
        </Alert>
      </Box>
    )
  }

  return (
    <MapContainer
      center={NYC_CENTER}
      zoom={DEFAULT_ZOOM}
      style={{ height: '100%', width: '100%', minHeight: '300px' }}
      key={theme.palette.mode} // Force re-render when theme changes
    >
      <TileLayer
        attribution={tileAttribution}
        url={tileUrl}
      />
      
      <MapBoundsUpdater bounds={mapBounds} onBoundsChange={handleBoundsChange} />
      
      <MarkerClusterGroupComponent locations={locations} />
    </MapContainer>
  )
}

export default MapView

