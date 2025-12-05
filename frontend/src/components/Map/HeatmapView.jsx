import { useEffect, useState, useRef } from 'react'
import { MapContainer, TileLayer, useMap } from 'react-leaflet'
import { useTheme, Box, Alert, Typography } from '@mui/material'
import L from 'leaflet'
import { getLocations } from '../../services/csvDataService'
import { MapSkeleton } from '../common/LoadingSkeleton'

function MapResizeHandler() {
  const map = useMap()
  
  useEffect(() => {
    const handleResize = () => {
      setTimeout(() => {
        map.invalidateSize()
      }, 100)
    }
    
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [map])
  
  return null
}

function HeatmapLayer({ locations }) {
  const map = useMap()
  const layerGroupRef = useRef(null)

  useEffect(() => {
    if (!layerGroupRef.current) {
      layerGroupRef.current = L.layerGroup().addTo(map)
    }

    // Clear existing layers
    layerGroupRef.current.clearLayers()

    // Find max count for normalization
    const maxCount = Math.max(
      ...locations.map(f => f.properties?.avg_recent_count || 0),
      1
    )

    // Create circle markers with varying sizes and colors based on count
    if (locations.length === 0) {
      return
    }

    locations.forEach(feature => {
      const { geometry, properties } = feature
      if (!geometry || !geometry.coordinates || !Array.isArray(geometry.coordinates)) return

      // GeoJSON uses [longitude, latitude], Leaflet uses [latitude, longitude]
      const [lng, lat] = geometry.coordinates
      
      // Validate coordinates
      if (typeof lat !== 'number' || typeof lng !== 'number' || 
          isNaN(lat) || isNaN(lng) ||
          lat < -90 || lat > 90 || lng < -180 || lng > 180) {
        console.warn('Invalid coordinates:', geometry.coordinates)
        return
      }

      const count = properties?.avg_recent_count || 0
      const normalized = count / maxCount

      // Calculate radius (5-30 pixels)
      const radius = Math.max(5, Math.min(30, 5 + normalized * 25))

      // Calculate color (blue -> green -> yellow -> red)
      let color = '#0000ff' // blue
      if (normalized > 0.75) color = '#ff0000' // red
      else if (normalized > 0.5) color = '#ffaa00' // orange
      else if (normalized > 0.25) color = '#ffff00' // yellow
      else color = '#00ff00' // green

      const circle = L.circleMarker([lat, lng], {
        radius,
        fillColor: color,
        color: '#fff',
        weight: 1,
        opacity: 0.8,
        fillOpacity: 0.6
      })

      circle.bindPopup(`
        <strong>${properties?.street_name_clean || properties?.street_clean || 'Unknown'}</strong><br/>
        Borough: ${properties?.borough || 'N/A'}<br/>
        Count: ${Math.round(count).toLocaleString()}
      `)

      layerGroupRef.current.addLayer(circle)
    })

    return () => {
      if (layerGroupRef.current) {
        layerGroupRef.current.clearLayers()
      }
    }
  }, [locations, map])

  return null
}

function HeatmapView({ filters }) {
  const theme = useTheme()
  const [locations, setLocations] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const fetchLocations = async () => {
      setLoading(true)
      try {
        // Build filters object for CSV service
        const filterParams = {}
        if (filters.boroughs?.length > 0) {
          filterParams.boroughs = filters.boroughs
        }
        if (filters.categories?.length > 0) {
          filterParams.categories = filters.categories
        }
        if (filters.minCount !== null && filters.minCount !== undefined) {
          filterParams.minCount = filters.minCount
        }
        if (filters.maxCount !== null && filters.maxCount !== undefined) {
          filterParams.maxCount = filters.maxCount
        }
        if (filters.search) {
          filterParams.search = filters.search
        }

        const data = await getLocations(filterParams)
        console.log('Heatmap data received:', data)
        
        if (data && data.features && Array.isArray(data.features)) {
          setLocations(data.features)
        } else if (data && Array.isArray(data)) {
          // Handle case where data is returned as array
          setLocations(data)
        } else {
          console.warn('No valid location data for heatmap')
          setLocations([])
        }
        setError(null)
      } catch (error) {
        console.error('Error fetching locations for heatmap:', error)
        setError('Failed to load heatmap data. Please try refreshing the page.')
        setLocations([])
      } finally {
        setLoading(false)
      }
    }

    fetchLocations()
  }, [filters])

  const tileUrl = theme.palette.mode === 'dark' 
    ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
    : 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'
  
  const tileAttribution = theme.palette.mode === 'dark'
    ? '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
    : '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'

  if (loading) {
    return <MapSkeleton />
  }

  if (error) {
    return (
      <Box sx={{ p: 2, height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Alert severity="error" sx={{ maxWidth: 500 }}>
          <Typography variant="h6" gutterBottom>Failed to load heatmap</Typography>
          <Typography variant="body2">{error}</Typography>
        </Alert>
      </Box>
    )
  }

  return (
    <MapContainer
      center={[40.7128, -73.9352]}
      zoom={11}
      style={{ height: '100%', width: '100%', minHeight: '300px' }}
      key={theme.palette.mode}
    >
      <TileLayer
        attribution={tileAttribution}
        url={tileUrl}
      />
      <MapResizeHandler />
      {locations.length > 0 && <HeatmapLayer locations={locations} />}
    </MapContainer>
  )
}

export default HeatmapView

