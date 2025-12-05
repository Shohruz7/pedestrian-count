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
  Grid,
  Card,
  CardContent,
  Tabs,
  Tab,
  Alert,
} from '@mui/material'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { getSummaryStats, getBoroughStats, getCategoryStats } from '../../services/csvDataService'
import { CardSkeleton, TableSkeleton, ChartSkeleton } from '../common/LoadingSkeleton'

function StatisticsPanel({ filters }) {
  const [summary, setSummary] = useState(null)
  const [boroughStats, setBoroughStats] = useState([])
  const [categoryStats, setCategoryStats] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [boroughChartTab, setBoroughChartTab] = useState(0)
  const [categoryChartTab, setCategoryChartTab] = useState(0)

  useEffect(() => {
    const fetchStats = async () => {
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

        const [summaryData, boroughData, categoryData] = await Promise.all([
          getSummaryStats(filterParams),
          getBoroughStats(),
          getCategoryStats(),
        ])

        setSummary(summaryData)
        setBoroughStats(boroughData.statistics || [])
        setCategoryStats(categoryData.statistics || [])
        setError(null)
      } catch (error) {
        console.error('Error fetching statistics:', error)
        setError('Failed to load statistics. Please try refreshing the page.')
      } finally {
        setLoading(false)
      }
    }

    fetchStats()
  }, [filters])

  if (loading) {
    return (
      <Box sx={{ p: 2 }}>
        <CardSkeleton />
        <Paper sx={{ mb: 3, p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Statistics by Borough
          </Typography>
          <TableSkeleton rows={6} cols={7} />
        </Paper>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Statistics by Category
          </Typography>
          <TableSkeleton rows={4} cols={7} />
        </Paper>
      </Box>
    )
  }

  if (error) {
    return (
      <Box sx={{ p: 2 }}>
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      </Box>
    )
  }

  return (
    <Box sx={{ p: 2 }}>
      {/* Summary Cards */}
      {summary && (
        <Grid container spacing={2} sx={{ mb: 4 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="textSecondary" gutterBottom>
                  Total Locations
                </Typography>
                <Typography variant="h4">
                  {summary.total_locations?.toLocaleString() || 0}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="textSecondary" gutterBottom>
                  Mean Count
                </Typography>
                <Typography variant="h4">
                  {Math.round(summary.mean_count || 0).toLocaleString()}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="textSecondary" gutterBottom>
                  Median Count
                </Typography>
                <Typography variant="h4">
                  {Math.round(summary.median_count || 0).toLocaleString()}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="textSecondary" gutterBottom>
                  Max Count
                </Typography>
                <Typography variant="h4">
                  {Math.round(summary.max_count || 0).toLocaleString()}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Borough Statistics */}
      <Paper sx={{ mb: 3, p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Statistics by Borough
        </Typography>
        <Tabs value={boroughChartTab} onChange={(e, v) => setBoroughChartTab(v)} sx={{ mb: 2 }}>
          <Tab label="Table" />
          <Tab label="Chart" />
        </Tabs>
        {boroughChartTab === 0 ? (
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Borough</TableCell>
                  <TableCell align="right">Locations</TableCell>
                  <TableCell align="right">Avg Count</TableCell>
                  <TableCell align="right">Median</TableCell>
                  <TableCell align="right">Min</TableCell>
                  <TableCell align="right">Max</TableCell>
                  <TableCell align="right">Std Dev</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {boroughStats.map((stat) => (
                  <TableRow key={stat.borough}>
                    <TableCell>{stat.borough}</TableCell>
                    <TableCell align="right">{stat.location_count}</TableCell>
                    <TableCell align="right">{Math.round(stat.avg_count || 0).toLocaleString()}</TableCell>
                    <TableCell align="right">{Math.round(stat.median_count || 0).toLocaleString()}</TableCell>
                    <TableCell align="right">{Math.round(stat.min_count || 0).toLocaleString()}</TableCell>
                    <TableCell align="right">{Math.round(stat.max_count || 0).toLocaleString()}</TableCell>
                    <TableCell align="right">{Math.round(stat.std_dev || 0).toLocaleString()}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        ) : (
          <Box sx={{ width: '100%', height: { xs: 300, md: 400 }, overflow: 'auto' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart 
                data={boroughStats.map(s => ({
                  ...s,
                  avg_count: Math.round(s.avg_count || 0),
                  median_count: Math.round(s.median_count || 0),
                  max_count: Math.round(s.max_count || 0)
                }))}
                margin={{ left: 20, right: 20, top: 20, bottom: 100 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="borough" angle={-45} textAnchor="end" height={100} />
                <YAxis 
                  width={80}
                  label={{ 
                    value: 'Pedestrian Count', 
                    angle: -90, 
                    position: 'left',
                    style: { textAnchor: 'middle' }
                  }} 
                />
                <Tooltip />
                <Legend />
                <Bar dataKey="avg_count" fill="#8884d8" name="Average Count" />
                <Bar dataKey="median_count" fill="#82ca9d" name="Median Count" />
                <Bar dataKey="max_count" fill="#ffc658" name="Max Count" />
              </BarChart>
            </ResponsiveContainer>
          </Box>
        )}
      </Paper>

      {/* Category Statistics */}
      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Statistics by Category
        </Typography>
        <Tabs value={categoryChartTab} onChange={(e, v) => setCategoryChartTab(v)} sx={{ mb: 2 }}>
          <Tab label="Table" />
          <Tab label="Chart" />
        </Tabs>
        {categoryChartTab === 0 ? (
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Category</TableCell>
                  <TableCell align="right">Locations</TableCell>
                  <TableCell align="right">Avg Count</TableCell>
                  <TableCell align="right">Median</TableCell>
                  <TableCell align="right">Min</TableCell>
                  <TableCell align="right">Max</TableCell>
                  <TableCell align="right">Std Dev</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {categoryStats.map((stat) => (
                  <TableRow key={stat.category}>
                    <TableCell>{stat.category}</TableCell>
                    <TableCell align="right">{stat.location_count}</TableCell>
                    <TableCell align="right">{Math.round(stat.avg_count || 0).toLocaleString()}</TableCell>
                    <TableCell align="right">{Math.round(stat.median_count || 0).toLocaleString()}</TableCell>
                    <TableCell align="right">{Math.round(stat.min_count || 0).toLocaleString()}</TableCell>
                    <TableCell align="right">{Math.round(stat.max_count || 0).toLocaleString()}</TableCell>
                    <TableCell align="right">{Math.round(stat.std_dev || 0).toLocaleString()}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        ) : (
          <Box sx={{ width: '100%', height: { xs: 300, md: 400 }, overflow: 'auto' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart 
                data={categoryStats.map(s => ({
                  ...s,
                  avg_count: Math.round(s.avg_count || 0),
                  median_count: Math.round(s.median_count || 0),
                  max_count: Math.round(s.max_count || 0)
                }))}
                margin={{ left: 20, right: 20, top: 20, bottom: 20 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="category" />
                <YAxis 
                  width={80}
                  label={{ 
                    value: 'Pedestrian Count', 
                    angle: -90, 
                    position: 'left',
                    style: { textAnchor: 'middle' }
                  }} 
                />
                <Tooltip />
                <Legend />
                <Bar dataKey="avg_count" fill="#8884d8" name="Average Count" />
                <Bar dataKey="median_count" fill="#82ca9d" name="Median Count" />
                <Bar dataKey="max_count" fill="#ffc658" name="Max Count" />
              </BarChart>
            </ResponsiveContainer>
          </Box>
        )}
      </Paper>
    </Box>
  )
}

export default StatisticsPanel

