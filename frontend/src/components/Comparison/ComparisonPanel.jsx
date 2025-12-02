import { useState } from 'react'
import {
  Box,
  Paper,
  Typography,
  FormControl,
  Select,
  MenuItem,
  Button,
  Grid,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  IconButton,
  Tooltip,
  Alert,
} from '@mui/material'
import { Close as CloseIcon, Info } from '@mui/icons-material'
import { compareGroups } from '../../services/api'

const BOROUGHS = ['The Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island', 'Bridges']
const CATEGORIES = ['Community', 'Global', 'Neighborhood', 'Regional']

function ComparisonPanel() {
  const [comparisonType, setComparisonType] = useState('borough')
  const [group1Values, setGroup1Values] = useState([])
  const [group2Values, setGroup2Values] = useState([])
  const [comparisonResult, setComparisonResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const options = comparisonType === 'borough' ? BOROUGHS : CATEGORIES

  const handleRemoveGroup1 = (valueToRemove, event) => {
    event?.stopPropagation()
    event?.preventDefault()
    setGroup1Values(group1Values.filter(v => v !== valueToRemove))
    setComparisonResult(null)
  }

  const handleRemoveGroup2 = (valueToRemove, event) => {
    event?.stopPropagation()
    event?.preventDefault()
    setGroup2Values(group2Values.filter(v => v !== valueToRemove))
    setComparisonResult(null)
  }

  const handleCompare = async () => {
    if (group1Values.length === 0 || group2Values.length === 0) {
      alert('Please select values for both groups')
      return
    }

    setLoading(true)
    try {
      const result = await compareGroups(
        {
          type: comparisonType,
          values: group1Values,
        },
        {
          type: comparisonType,
          values: group2Values,
        }
      )
      setComparisonResult(result)
      setError(null)
    } catch (error) {
      console.error('Error comparing groups:', error)
      setError('Failed to compare groups. Please check your selections and try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Box sx={{ p: 2 }}>
      <Paper sx={{ p: 3, mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
          <Typography variant="h6">
            Comparison Mode
          </Typography>
          <Tooltip title="Compare statistics between different groups (boroughs or categories). Select values for Group 1 and Group 2, then click Compare to see side-by-side metrics and differences.">
            <Info fontSize="small" color="action" />
          </Tooltip>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 1 }}>
              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                Comparison Type
              </Typography>
              <Tooltip title="Choose whether to compare by Borough or Category">
                <Info fontSize="small" color="action" sx={{ fontSize: 14 }} />
              </Tooltip>
            </Box>
            <FormControl fullWidth>
              <Select
                value={comparisonType}
                onChange={(e) => {
                  setComparisonType(e.target.value)
                  setGroup1Values([])
                  setGroup2Values([])
                  setComparisonResult(null)
                }}
                displayEmpty
              >
                <MenuItem value="borough">Borough</MenuItem>
                <MenuItem value="category">Category</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={4}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 1 }}>
              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                Group 1
              </Typography>
              <Tooltip title="Select one or more values for the first comparison group">
                <Info fontSize="small" color="action" sx={{ fontSize: 14 }} />
              </Tooltip>
            </Box>
            <FormControl fullWidth>
              <Select
                multiple
                value={group1Values}
                onChange={(e) => setGroup1Values(e.target.value)}
                displayEmpty
                renderValue={(selected) => {
                  if (selected.length === 0) {
                    return <em>Select values</em>
                  }
                  return (
                    <Box 
                      onMouseDown={(e) => e.stopPropagation()}
                      onClick={(e) => e.stopPropagation()}
                      sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, pointerEvents: 'auto' }}
                    >
                      {selected.map((value) => (
                        <Box
                          key={value}
                          onMouseDown={(e) => e.stopPropagation()}
                          onMouseEnter={(e) => e.stopPropagation()}
                          onMouseLeave={(e) => e.stopPropagation()}
                          onClick={(e) => e.stopPropagation()}
                          sx={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: 0.5,
                            backgroundColor: 'action.selected',
                            borderRadius: 1,
                            px: 1,
                            py: 0.5,
                            fontSize: '0.875rem',
                          }}
                        >
                          <span>{value}</span>
                          <IconButton
                            size="small"
                            onMouseDown={(e) => {
                              e.stopPropagation()
                              e.preventDefault()
                            }}
                            onMouseEnter={(e) => e.stopPropagation()}
                            onMouseLeave={(e) => e.stopPropagation()}
                            onClick={(e) => {
                              e.stopPropagation()
                              e.preventDefault()
                              handleRemoveGroup1(value, e)
                            }}
                            sx={{ 
                              p: 0.25,
                              width: 18,
                              height: 18,
                              '&:hover': { backgroundColor: 'rgba(0, 0, 0, 0.1)' }
                            }}
                          >
                            <CloseIcon sx={{ fontSize: 12 }} />
                          </IconButton>
                        </Box>
                      ))}
                    </Box>
                  )
                }}
              >
                {options.map((option) => (
                  <MenuItem key={option} value={option}>
                    {option}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={4}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 1 }}>
              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                Group 2
              </Typography>
              <Tooltip title="Select one or more values for the second comparison group">
                <Info fontSize="small" color="action" sx={{ fontSize: 14 }} />
              </Tooltip>
            </Box>
            <FormControl fullWidth>
              <Select
                multiple
                value={group2Values}
                onChange={(e) => setGroup2Values(e.target.value)}
                displayEmpty
                renderValue={(selected) => {
                  if (selected.length === 0) {
                    return <em>Select values</em>
                  }
                  return (
                    <Box 
                      onMouseDown={(e) => e.stopPropagation()}
                      onClick={(e) => e.stopPropagation()}
                      sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, pointerEvents: 'auto' }}
                    >
                      {selected.map((value) => (
                        <Box
                          key={value}
                          onMouseDown={(e) => e.stopPropagation()}
                          onMouseEnter={(e) => e.stopPropagation()}
                          onMouseLeave={(e) => e.stopPropagation()}
                          onClick={(e) => e.stopPropagation()}
                          sx={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: 0.5,
                            backgroundColor: 'action.selected',
                            borderRadius: 1,
                            px: 1,
                            py: 0.5,
                            fontSize: '0.875rem',
                          }}
                        >
                          <span>{value}</span>
                          <IconButton
                            size="small"
                            onMouseDown={(e) => {
                              e.stopPropagation()
                              e.preventDefault()
                            }}
                            onMouseEnter={(e) => e.stopPropagation()}
                            onMouseLeave={(e) => e.stopPropagation()}
                            onClick={(e) => {
                              e.stopPropagation()
                              e.preventDefault()
                              handleRemoveGroup2(value, e)
                            }}
                            sx={{ 
                              p: 0.25,
                              width: 18,
                              height: 18,
                              '&:hover': { backgroundColor: 'rgba(0, 0, 0, 0.1)' }
                            }}
                          >
                            <CloseIcon sx={{ fontSize: 12 }} />
                          </IconButton>
                        </Box>
                      ))}
                    </Box>
                  )
                }}
              >
                {options.map((option) => (
                  <MenuItem key={option} value={option}>
                    {option}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
        </Grid>

        <Button
          variant="contained"
          onClick={handleCompare}
          disabled={loading || group1Values.length === 0 || group2Values.length === 0}
          sx={{ mt: 2 }}
        >
          {loading ? 'Comparing...' : 'Compare'}
        </Button>
      </Paper>

      {comparisonResult && (
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Group 1
                </Typography>
                <Typography>Count: {comparisonResult.group1.statistics.count}</Typography>
                <Typography>Mean: {Math.round(comparisonResult.group1.statistics.mean).toLocaleString()}</Typography>
                <Typography>Median: {Math.round(comparisonResult.group1.statistics.median).toLocaleString()}</Typography>
                <Typography>Max: {Math.round(comparisonResult.group1.statistics.max).toLocaleString()}</Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Group 2
                </Typography>
                <Typography>Count: {comparisonResult.group2.statistics.count}</Typography>
                <Typography>Mean: {Math.round(comparisonResult.group2.statistics.mean).toLocaleString()}</Typography>
                <Typography>Median: {Math.round(comparisonResult.group2.statistics.median).toLocaleString()}</Typography>
                <Typography>Max: {Math.round(comparisonResult.group2.statistics.max).toLocaleString()}</Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Differences
              </Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Metric</TableCell>
                      <TableCell align="right">Absolute</TableCell>
                      <TableCell align="right">Percentage</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    <TableRow>
                      <TableCell>Count</TableCell>
                      <TableCell align="right">
                        {comparisonResult.differences.count.absolute}
                      </TableCell>
                      <TableCell align="right">
                        {comparisonResult.differences.count.percentage}%
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Mean</TableCell>
                      <TableCell align="right">
                        {Math.round(comparisonResult.differences.mean.absolute).toLocaleString()}
                      </TableCell>
                      <TableCell align="right">
                        {comparisonResult.differences.mean.percentage}%
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Median</TableCell>
                      <TableCell align="right">
                        {Math.round(comparisonResult.differences.median.absolute).toLocaleString()}
                      </TableCell>
                      <TableCell align="right">
                        {comparisonResult.differences.median.percentage}%
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Max</TableCell>
                      <TableCell align="right">
                        {Math.round(comparisonResult.differences.max.absolute).toLocaleString()}
                      </TableCell>
                      <TableCell align="right">
                        {comparisonResult.differences.max.percentage}%
                      </TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </TableContainer>
            </Paper>
          </Grid>
        </Grid>
      )}
    </Box>
  )
}

export default ComparisonPanel

