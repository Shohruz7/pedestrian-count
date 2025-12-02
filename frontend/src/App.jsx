import { useState, useMemo, useEffect } from 'react'
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material'
import MapView from './components/Map/MapView'
import HeatmapView from './components/Map/HeatmapView'
import FilterPanel from './components/Filters/FilterPanel'
import StatisticsPanel from './components/Statistics/StatisticsPanel'
import ComparisonPanel from './components/Comparison/ComparisonPanel'
import TimeSeriesPanel from './components/TimeSeries/TimeSeriesPanel'
import TopSitesPanel from './components/TopSites/TopSitesPanel'
import ExportPanel from './components/Export/ExportPanel'
import ErrorBoundary from './components/common/ErrorBoundary'
import { Box, Tabs, Tab, Paper, IconButton, Tooltip, ToggleButton, ToggleButtonGroup, Typography, Chip } from '@mui/material'
import { Brightness4, Brightness7, Print, Refresh } from '@mui/icons-material'
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts'

function TabPanel({ children, value, index }) {
  return (
    <div role="tabpanel" hidden={value !== index}>
      {value === index && <Box sx={{ p: { xs: 1, md: 3 } }}>{children}</Box>}
    </div>
  )
}

function App() {
  const [mode, setMode] = useState(() => {
    // Check localStorage for saved preference, default to 'light'
    const savedMode = localStorage.getItem('themeMode')
    return savedMode || 'light'
  })
  const [tabValue, setTabValue] = useState(0)
  const [mapViewMode, setMapViewMode] = useState('markers') // 'markers' or 'heatmap'
  const [lastRefresh, setLastRefresh] = useState(new Date())
  const [filters, setFilters] = useState(() => {
    // Load filters from URL parameters if present
    const params = new URLSearchParams(window.location.search)
    const urlFilters = {
      boroughs: params.getAll('borough') || [],
      categories: params.getAll('category') || [],
      minCount: params.get('minCount') ? Number(params.get('minCount')) : null,
      maxCount: params.get('maxCount') ? Number(params.get('maxCount')) : null,
      search: params.get('search') || '',
      dateRange: null,
    }
    return urlFilters
  })

  // Create theme based on mode
  const theme = useMemo(
    () =>
      createTheme({
        palette: {
          mode: mode,
          primary: {
            main: '#1976d2',
          },
          secondary: {
            main: '#dc004e',
          },
        },
      }),
    [mode]
  )

  const toggleColorMode = () => {
    const newMode = mode === 'light' ? 'dark' : 'light'
    setMode(newMode)
    localStorage.setItem('themeMode', newMode)
  }

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue)
  }

  const handleRefresh = () => {
    setLastRefresh(new Date())
    // Force a re-render by updating a key or triggering a refresh
    window.location.reload()
  }

  // Keyboard shortcuts
  useKeyboardShortcuts([
    { key: '1', handler: () => setTabValue(0), description: 'Go to Map & Filters' },
    { key: '2', handler: () => setTabValue(1), description: 'Go to Comparison' },
    { key: '3', handler: () => setTabValue(2), description: 'Go to Statistics' },
    { key: '4', handler: () => setTabValue(3), description: 'Go to Time Series' },
    { key: '5', handler: () => setTabValue(4), description: 'Go to Top Sites' },
    { key: '6', handler: () => setTabValue(5), description: 'Go to Export' },
    { key: 'ctrl+r', handler: (e) => { e.preventDefault(); handleRefresh() }, description: 'Refresh data' },
    { key: 'ctrl+/', handler: () => alert('Keyboard Shortcuts:\n1-6: Switch tabs\nCtrl+R: Refresh\nCtrl+/: Show this help'), description: 'Show shortcuts' },
  ])

  useEffect(() => {
    // Update last refresh time every minute
    const interval = setInterval(() => {
      setLastRefresh(new Date())
    }, 60000)
    return () => clearInterval(interval)
  }, [])

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <ErrorBoundary>
        <Box sx={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
        {/* Header */}
        <Paper elevation={2} sx={{ p: { xs: 1, md: 2 }, position: 'relative' }}>
          <Box sx={{ 
            display: 'flex', 
            flexDirection: { xs: 'column', sm: 'row' },
            justifyContent: 'space-between', 
            alignItems: { xs: 'flex-start', sm: 'flex-start' },
            gap: { xs: 1, sm: 0 }
          }}>
            <Box sx={{ flex: 1 }}>
              <Typography variant="h4" component="h1" sx={{ fontSize: { xs: '1.5rem', md: '2rem' }, margin: 0 }}>
                NYC Pedestrian Count Dashboard
              </Typography>
              <Typography variant="body2" sx={{ margin: '8px 0 0 0', color: 'text.secondary' }}>
                Visualizing pedestrian count data from NYC DOT's Bi-Annual Pedestrian Counts
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', gap: 1, mt: { xs: 1, sm: 0 }, alignItems: 'center' }}>
              <Chip
                icon={<Refresh />}
                label={`Updated ${lastRefresh.toLocaleTimeString()}`}
                size="small"
                variant="outlined"
                sx={{ display: { xs: 'none', sm: 'flex' } }}
              />
              <Tooltip title="Refresh data (Ctrl+R)">
                <IconButton
                  onClick={handleRefresh}
                  color="inherit"
                  aria-label="refresh"
                  size={window.innerWidth < 600 ? 'small' : 'medium'}
                >
                  <Refresh />
                </IconButton>
              </Tooltip>
              <Tooltip title="Print view">
                <IconButton
                  onClick={() => window.print()}
                  color="inherit"
                  aria-label="print"
                  className="no-print"
                  size={window.innerWidth < 600 ? 'small' : 'medium'}
                >
                  <Print />
                </IconButton>
              </Tooltip>
              <Tooltip title={mode === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}>
                <IconButton
                  onClick={toggleColorMode}
                  color="inherit"
                  aria-label="toggle theme"
                  size={window.innerWidth < 600 ? 'small' : 'medium'}
                >
                  {mode === 'light' ? <Brightness4 /> : <Brightness7 />}
                </IconButton>
              </Tooltip>
            </Box>
          </Box>
        </Paper>

        {/* Tabs */}
        <Paper square className="no-print">
          <Tabs 
            value={tabValue} 
            onChange={handleTabChange}
            variant="scrollable"
            scrollButtons="auto"
            sx={{ 
              '& .MuiTab-root': {
                minWidth: { xs: 80, sm: 120 },
                fontSize: { xs: '0.75rem', sm: '0.875rem' },
                px: { xs: 1, sm: 2 }
              }
            }}
          >
            <Tab label="Map & Filters" />
            <Tab label="Comparison" />
            <Tab label="Statistics" />
            <Tab label="Time Series" />
            <Tab label="Top Sites" />
            <Tab label="Export" />
          </Tabs>
        </Paper>

        {/* Tab Content */}
        <Box sx={{ flex: 1, overflow: 'auto' }}>
          <TabPanel value={tabValue} index={0}>
            <Box sx={{ 
              display: 'flex', 
              flexDirection: { xs: 'column', md: 'row' },
              height: { xs: 'auto', md: 'calc(100vh - 200px)' },
              gap: 2,
              p: { xs: 1, md: 0 }
            }}>
              <Box sx={{ 
                width: { xs: '100%', md: '300px' },
                minWidth: { xs: '100%', md: '300px' },
                maxWidth: { xs: '100%', md: '300px' },
                overflow: 'auto',
                height: { xs: '300px', md: 'auto' }
              }}>
                <FilterPanel filters={filters} setFilters={setFilters} />
              </Box>
              <Box sx={{ 
                flex: 1, 
                display: 'flex', 
                flexDirection: 'column',
                minHeight: { xs: '400px', md: 'auto' },
                height: { xs: '400px', md: 'auto' }
              }}>
                <Box sx={{ 
                  mb: 1, 
                  display: 'flex', 
                  justifyContent: 'flex-end',
                  px: { xs: 1, md: 0 }
                }}>
                  <ToggleButtonGroup
                    value={mapViewMode}
                    exclusive
                    onChange={(e, newMode) => newMode && setMapViewMode(newMode)}
                    size="small"
                  >
                    <ToggleButton value="markers">Markers</ToggleButton>
                    <ToggleButton value="heatmap">Heatmap</ToggleButton>
                  </ToggleButtonGroup>
                </Box>
                <Box sx={{ flex: 1, minHeight: 0 }}>
                  {mapViewMode === 'markers' ? (
                    <MapView filters={filters} />
                  ) : (
                    <HeatmapView filters={filters} />
                  )}
                </Box>
              </Box>
            </Box>
          </TabPanel>

          <TabPanel value={tabValue} index={1}>
            <ComparisonPanel />
          </TabPanel>

          <TabPanel value={tabValue} index={2}>
            <StatisticsPanel filters={filters} />
          </TabPanel>

          <TabPanel value={tabValue} index={3}>
            <TimeSeriesPanel />
          </TabPanel>

          <TabPanel value={tabValue} index={4}>
            <TopSitesPanel filters={filters} />
          </TabPanel>

          <TabPanel value={tabValue} index={5}>
            <ExportPanel filters={filters} />
          </TabPanel>
        </Box>
      </Box>
      </ErrorBoundary>
    </ThemeProvider>
  )
}

export default App

