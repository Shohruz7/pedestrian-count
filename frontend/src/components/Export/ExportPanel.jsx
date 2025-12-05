import { useState } from 'react'
import {
  Box,
  Paper,
  Typography,
  Button,
  ButtonGroup,
  CircularProgress,
} from '@mui/material'
import { Download as DownloadIcon } from '@mui/icons-material'
import { exportCSV, exportGeoJSON } from '../../services/csvDataService'

function ExportPanel({ filters }) {
  const [loading, setLoading] = useState(false)

  const handleExport = async (format) => {
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
      if (filters.minCount !== null) {
        filterParams.minCount = filters.minCount
      }
      if (filters.maxCount !== null) {
        filterParams.maxCount = filters.maxCount
      }
      if (filters.search) {
        filterParams.search = filters.search
      }

      let blob, filename
      if (format === 'csv') {
        blob = await exportCSV(filterParams)
        filename = 'pedestrian_data.csv'
      } else {
        blob = await exportGeoJSON(filterParams)
        filename = 'pedestrian_data.geojson'
      }

      // Create download link
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = filename
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (error) {
      console.error('Error exporting data:', error)
      alert('Error exporting data. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Box sx={{ p: 2 }}>
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Export Data
        </Typography>
        <Typography variant="body2" color="textSecondary" sx={{ mb: 3 }}>
          Export filtered data in various formats. Current filters will be applied to the export.
        </Typography>

        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
          <Button
            variant="contained"
            startIcon={loading ? <CircularProgress size={20} /> : <DownloadIcon />}
            onClick={() => handleExport('csv')}
            disabled={loading}
            size="large"
          >
            Download CSV
          </Button>
          <Button
            variant="contained"
            startIcon={loading ? <CircularProgress size={20} /> : <DownloadIcon />}
            onClick={() => handleExport('geojson')}
            disabled={loading}
            size="large"
          >
            Download GeoJSON
          </Button>
        </Box>
      </Paper>
    </Box>
  )
}

export default ExportPanel


