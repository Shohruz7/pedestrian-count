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
} from '@mui/material'
import { Info } from '@mui/icons-material'
import { getTopSites } from '../../services/api'
import { TableSkeleton } from '../common/LoadingSkeleton'

function TopSitesPanel({ filters }) {
  const [sites, setSites] = useState([])
  const [limit, setLimit] = useState(10)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    const fetchTopSites = async () => {
      setLoading(true)
      try {
        const data = await getTopSites(limit)
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
  }, [limit])

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
          <TextField
            label="Number of Sites"
            type="number"
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value))}
            inputProps={{ min: 1, max: 100 }}
            size="small"
            sx={{ width: { xs: '100%', sm: 150 } }}
          />
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

