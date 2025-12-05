import { useState, useEffect } from 'react'
import {
  Box,
  Paper,
  Typography,
  FormControl,
  Select,
  MenuItem,
  Alert,
  Skeleton,
} from '@mui/material'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { getTimeSeries, getLocations } from '../../services/csvDataService'
import { ChartSkeleton } from '../common/LoadingSkeleton'

function TimeSeriesPanel() {
  const [locations, setLocations] = useState([])
  const [selectedLocationId, setSelectedLocationId] = useState(null)
  const [timeSeriesData, setTimeSeriesData] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [loadingLocations, setLoadingLocations] = useState(true)

  useEffect(() => {
    const fetchLocations = async () => {
      setLoadingLocations(true)
      try {
        const data = await getLocations({})
        // csvDataService returns GeoJSON FeatureCollection
        const locs = (data.features || []).map(f => ({
          id: f.properties?.objectid || f.properties?.OBJECTID || f.properties?.id,
          objectid: f.properties?.objectid || f.properties?.OBJECTID,
          loc_id: f.properties?.loc_id || f.properties?.Loc,
          borough: f.properties?.borough || f.properties?.Borough,
          street_name_clean: f.properties?.street_name_clean || f.properties?.Street_Nam_clean,
          category: f.properties?.category || f.properties?.Category,
        }))
        setLocations(locs)
        if (locs.length > 0) {
          setSelectedLocationId(locs[0].id)
        }
        setError(null)
      } catch (error) {
        console.error('Error fetching locations:', error)
        setError('Failed to load locations. Please try refreshing the page.')
      } finally {
        setLoadingLocations(false)
      }
    }
    fetchLocations()
  }, [])

  useEffect(() => {
    if (selectedLocationId) {
      const fetchTimeSeries = async () => {
        setLoading(true)
        try {
          const data = await getTimeSeries(selectedLocationId)
          setTimeSeriesData(data.counts || [])
        } catch (error) {
          console.error('Error fetching time series:', error)
        } finally {
          setLoading(false)
        }
      }
      fetchTimeSeries()
    }
  }, [selectedLocationId])

  // Transform data for chart - group by date, separate by period
  const chartData = timeSeriesData.reduce((acc, item) => {
    const date = item.count_date
    const period = item.period
    const countValue = Math.round(item.count_value || 0) // Round to whole number
    
    if (!acc[date]) {
      acc[date] = { date }
    }
    acc[date][period] = countValue
    
    return acc
  }, {})

  const chartDataArray = Object.values(chartData)
    .sort((a, b) => new Date(a.date) - new Date(b.date))
    .map(item => ({
      ...item,
      date: new Date(item.date).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })
    }))

  return (
    <Box sx={{ p: 2 }}>
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Time Series Analysis
        </Typography>


        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {loadingLocations ? (
          <Box sx={{ mb: 3, maxWidth: 400 }}>
            <Skeleton variant="text" width="30%" height={24} sx={{ mb: 1 }} />
            <Skeleton variant="rectangular" width="100%" height={56} />
          </Box>
        ) : (
          <Box sx={{ mb: 3, maxWidth: 400 }}>
            <Typography variant="body2" sx={{ mb: 1, fontWeight: 500 }}>
              Select Location
            </Typography>
            <FormControl fullWidth>
              <Select
                value={selectedLocationId || ''}
                onChange={(e) => setSelectedLocationId(e.target.value)}
                displayEmpty
              >
                <MenuItem value="" disabled>
                  <em>Select a location</em>
                </MenuItem>
                {locations.map((loc) => (
                  <MenuItem key={loc.id} value={loc.id}>
                    {loc.street_name_clean || loc.street_clean || `Location ${loc.id}`}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Box>
        )}

        {loading ? (
          <ChartSkeleton height={400} />
        ) : chartDataArray.length > 0 ? (
          <Box>
            <Box sx={{ mb: 2, p: 1.5, bgcolor: 'background.default', borderRadius: 1, border: 1, borderColor: 'divider' }}>
              <Typography variant="body2" sx={{ fontWeight: 500, mb: 0.5 }}>
                Period Definitions:
              </Typography>
              <Typography variant="body2" component="div">
                <strong>AM:</strong> Morning period (typically 7:00 AM - 9:00 AM)
                <br />
                <strong>PM:</strong> Afternoon/Evening period (typically 4:00 PM - 7:00 PM)
                <br />
                <strong>MD:</strong> Midday period (typically 12:00 PM - 2:00 PM)
              </Typography>
            </Box>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={chartDataArray}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis 
                  label={{ 
                    value: 'Pedestrian Count', 
                    angle: -90, 
                    position: 'insideLeft',
                    style: { textAnchor: 'middle' }
                  }} 
                />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="AM" stroke="#8884d8" name="AM (Morning)" />
                <Line type="monotone" dataKey="PM" stroke="#82ca9d" name="PM (Evening)" />
                <Line type="monotone" dataKey="MD" stroke="#ffc658" name="MD (Midday)" />
              </LineChart>
            </ResponsiveContainer>
          </Box>
        ) : (
          <Typography>No time series data available for this location</Typography>
        )}
      </Paper>
    </Box>
  )
}

export default TimeSeriesPanel

