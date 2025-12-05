import { useState, useEffect } from 'react'
import {
  Box,
  Paper,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Alert,
  Tooltip,
  FormControl,
  Select,
  MenuItem,
  InputLabel,
} from '@mui/material'
import { Info } from '@mui/icons-material'
import { getTopSites, getSiteCountByBorough } from '../../services/csvDataService'
import { TableSkeleton } from '../common/LoadingSkeleton'

const BOROUGHS = ['The Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island', 'Bridges']

function TopSitesPanel({ filters }) {
  const [sites, setSites] = useState([])
  const [limit, setLimit] = useState(10)
  const [selectedBorough, setSelectedBorough] = useState('')
  const [maxSites, setMaxSites] = useState(100) // Default max, will be updated
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Fetch max sites count when borough changes
  useEffect(() => {
    const fetchMaxSites = async () => {
      try {
        const boroughFilter = selectedBorough === '' ? null : selectedBorough
        const count = await getSiteCountByBorough(boroughFilter)
        setMaxSites(count)
        // If current limit exceeds max, adjust it
        if (limit > count) {
          setLimit(count)
        }
      } catch (error) {
        console.error('Error fetching max sites count:', error)
        // Keep default max on error
      }
    }

    fetchMaxSites()
  }, [selectedBorough])

  useEffect(() => {
    const fetchTopSites = async () => {
      setLoading(true)
      try {
        // Pass empty string or null if no borough selected (shows all)
        const boroughFilter = selectedBorough === '' ? null : selectedBorough
        const data = await getTopSites(limit, boroughFilter)
        setSites(data.sites || [])
        setError(null)
      } catch (error) {
        console.error('Error fetching top sites:', error)
        setError('Failed to load top sites. Please try again.')
      } finally {
        setLoading(false)
      }
    }

    fetchTopSites()
  }, [limit, selectedBorough])

  return (
    <Box sx={{ p: 2 }}>
      <Paper sx={{ p: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2, flexWrap: 'wrap', gap: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography variant="h6">
              Top Sites by Pedestrian Count
            </Typography>
            <Tooltip title="Shows locations with the highest average pedestrian counts. Rankings are based on recent count data.">
              <Info fontSize="small" color="action" />
            </Tooltip>
          </Box>
          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'center' }}>
            <FormControl size="small" sx={{ minWidth: { xs: '100%', sm: 180 }, verticalAlign: 'middle' }}>
              <InputLabel>Filter by Borough</InputLabel>
              <Select
                value={selectedBorough}
                label="Filter by Borough"
                onChange={(e) => setSelectedBorough(e.target.value)}
              >
                <MenuItem value="">All Boroughs</MenuItem>
                {BOROUGHS.map((borough) => (
                  <MenuItem key={borough} value={borough}>
                    {borough}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <TextField
              label="Number of Sites"
              type="number"
              value={limit}
              onChange={(e) => {
                const newValue = Number(e.target.value)
                // Ensure value doesn't exceed max
                const clampedValue = Math.min(Math.max(1, newValue), maxSites)
                setLimit(clampedValue)
              }}
              inputProps={{ min: 1, max: maxSites }}
              size="small"
              sx={{ width: { xs: '100%', sm: 150 }, '& .MuiFormHelperText-root': { position: 'absolute', bottom: -20 } }}
              helperText={`Max: ${maxSites} sites`}
            />
          </Box>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {loading ? (
          <TableSkeleton rows={limit} cols={5} />
        ) : (
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Rank</TableCell>
                  <TableCell>Street Name</TableCell>
                  <TableCell>Borough</TableCell>
                  <TableCell>Category</TableCell>
                  <TableCell align="right">Avg Count</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {sites.map((site, index) => (
                  <TableRow key={site.location?.id || index}>
                    <TableCell>{index + 1}</TableCell>
                    <TableCell>
                      {site.location?.street_name_clean || site.location?.street_clean || 'Unknown'}
                    </TableCell>
                    <TableCell>{site.location?.borough || 'Unknown'}</TableCell>
                    <TableCell>{site.location?.category || 'Unknown'}</TableCell>
                    <TableCell align="right">
                      {Math.round(site.avg_recent_count || 0).toLocaleString()}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </Paper>
    </Box>
  )
}

export default TopSitesPanel

