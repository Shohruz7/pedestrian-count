import { useState, useEffect } from 'react'
import {
  Box,
  Paper,
  Typography,
  Checkbox,
  FormControlLabel,
  FormGroup,
  TextField,
  Button,
  Divider,
  IconButton,
  Menu,
  MenuItem,
  ListItemText,
  Tooltip,
} from '@mui/material'
import { Save, Share, GetApp } from '@mui/icons-material'
import { getLocations } from '../../services/csvDataService'

const BOROUGHS = ['The Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island', 'Bridges']
const CATEGORIES = ['Community', 'Global', 'Neighborhood', 'Regional']

function FilterPanel({ filters, setFilters }) {
  const [availableBoroughs, setAvailableBoroughs] = useState(BOROUGHS)
  const [availableCategories, setAvailableCategories] = useState(CATEGORIES)
  const [savedFiltersMenu, setSavedFiltersMenu] = useState(null)
  const [savedFilters, setSavedFilters] = useState(() => {
    const saved = localStorage.getItem('savedFilters')
    return saved ? JSON.parse(saved) : []
  })

  const handleBoroughChange = (borough) => {
    const newBoroughs = filters.boroughs?.includes(borough)
      ? filters.boroughs.filter(b => b !== borough)
      : [...(filters.boroughs || []), borough]
    setFilters({ ...filters, boroughs: newBoroughs })
  }

  const handleCategoryChange = (category) => {
    const newCategories = filters.categories?.includes(category)
      ? filters.categories.filter(c => c !== category)
      : [...(filters.categories || []), category]
    setFilters({ ...filters, categories: newCategories })
  }

  const handleCountRangeChange = (field, value) => {
    setFilters({ ...filters, [field]: value })
  }

  const handleSearchChange = (event) => {
    setFilters({ ...filters, search: event.target.value })
  }

  const handleClearFilters = () => {
    setFilters({
      boroughs: [],
      categories: [],
      minCount: null,
      maxCount: null,
      search: '',
      dateRange: null,
    })
  }

  const handleSaveFilters = () => {
    const name = prompt('Enter a name for this filter preset:')
    if (name && name.trim()) {
      const newSaved = [...savedFilters, { name: name.trim(), filters: { ...filters } }]
      setSavedFilters(newSaved)
      localStorage.setItem('savedFilters', JSON.stringify(newSaved))
    }
  }

  const handleLoadFilter = (savedFilter) => {
    setFilters(savedFilter.filters)
    setSavedFiltersMenu(null)
  }

  const handleDeleteFilter = (index, e) => {
    e.stopPropagation()
    const newSaved = savedFilters.filter((_, i) => i !== index)
    setSavedFilters(newSaved)
    localStorage.setItem('savedFilters', JSON.stringify(newSaved))
  }

  const handleShare = () => {
    const params = new URLSearchParams()
    if (filters.boroughs?.length > 0) {
      filters.boroughs.forEach(b => params.append('borough', b))
    }
    if (filters.categories?.length > 0) {
      filters.categories.forEach(c => params.append('category', c))
    }
    if (filters.minCount !== null) params.set('minCount', filters.minCount)
    if (filters.maxCount !== null) params.set('maxCount', filters.maxCount)
    if (filters.search) params.set('search', filters.search)
    
    const url = `${window.location.origin}${window.location.pathname}?${params.toString()}`
    navigator.clipboard.writeText(url).then(() => {
      alert('Shareable URL copied to clipboard!')
    }).catch(() => {
      // Fallback
      prompt('Copy this URL:', url)
    })
  }

  return (
    <Paper sx={{ p: 2, height: '100%', overflow: 'auto' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">
          Filters
        </Typography>
        <Box>
          <Tooltip title="Save current filter settings for quick access later">
            <IconButton size="small" onClick={handleSaveFilters}>
              <Save fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Load a previously saved filter preset">
            <IconButton 
              size="small" 
              onClick={(e) => setSavedFiltersMenu(e.currentTarget)}
            >
              <GetApp fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Generate a shareable URL with current filter settings">
            <IconButton size="small" onClick={handleShare}>
              <Share fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      <Menu
        anchorEl={savedFiltersMenu}
        open={Boolean(savedFiltersMenu)}
        onClose={() => setSavedFiltersMenu(null)}
      >
        {savedFilters.length === 0 ? (
          <MenuItem disabled>No saved filters</MenuItem>
        ) : (
          savedFilters.map((saved, index) => (
            <MenuItem 
              key={index}
              onClick={() => handleLoadFilter(saved)}
              sx={{ display: 'flex', justifyContent: 'space-between', minWidth: 200 }}
            >
              <ListItemText primary={saved.name} />
              <IconButton 
                size="small" 
                onClick={(e) => handleDeleteFilter(index, e)}
                sx={{ ml: 1 }}
              >
                <Typography variant="body2">×</Typography>
              </IconButton>
            </MenuItem>
          ))
        )}
      </Menu>

      {/* Search */}
      <Box sx={{ mb: 2 }}>
        <TextField
          fullWidth
          label="Search (Street or Location ID)"
          variant="outlined"
          size="small"
          value={filters.search || ''}
          onChange={handleSearchChange}
          placeholder="Search..."
        />
      </Box>

      <Divider sx={{ my: 2 }} />

      {/* Boroughs */}
      <Typography variant="subtitle2" gutterBottom>
        Boroughs and Bridges
      </Typography>
      <FormGroup>
        {availableBoroughs.map((borough) => (
          <FormControlLabel
            key={borough}
            control={
              <Checkbox
                checked={filters.boroughs?.includes(borough) || false}
                onChange={() => handleBoroughChange(borough)}
                size="small"
              />
            }
            label={borough}
          />
        ))}
      </FormGroup>

      <Divider sx={{ my: 2 }} />

      {/* Categories */}
      <Typography variant="subtitle2" gutterBottom>
        Categories
      </Typography>
      <FormGroup>
        {availableCategories.map((category) => (
          <FormControlLabel
            key={category}
            control={
              <Checkbox
                checked={filters.categories?.includes(category) || false}
                onChange={() => handleCategoryChange(category)}
                size="small"
              />
            }
            label={category}
          />
        ))}
      </FormGroup>

      <Divider sx={{ my: 2 }} />

      {/* Count Range */}
      <Typography variant="subtitle2" gutterBottom>
        Count Range
      </Typography>
      <Box sx={{ px: 1 }}>
        <TextField
          fullWidth
          label="Min Count"
          type="number"
          size="small"
          value={filters.minCount || ''}
          onChange={(e) => handleCountRangeChange('minCount', e.target.value ? Number(e.target.value) : null)}
          sx={{ mb: 1 }}
        />
        <TextField
          fullWidth
          label="Max Count"
          type="number"
          size="small"
          value={filters.maxCount || ''}
          onChange={(e) => handleCountRangeChange('maxCount', e.target.value ? Number(e.target.value) : null)}
        />
      </Box>

      <Divider sx={{ my: 2 }} />

      {/* Date Range (for time series filtering) */}
      <Typography variant="subtitle2" gutterBottom sx={{ mb: 1.5 }}>
        Date Range (Optional)
      </Typography>
      <Box sx={{ px: 1 }}>
        <TextField
          fullWidth
          label="Start Date"
          type="date"
          size="small"
          value={filters.dateRange?.start || ''}
          onChange={(e) => setFilters({
            ...filters,
            dateRange: { ...filters.dateRange, start: e.target.value }
          })}
          InputLabelProps={{ shrink: true }}
          sx={{ mb: 2 }}
        />
        <TextField
          fullWidth
          label="End Date"
          type="date"
          size="small"
          value={filters.dateRange?.end || ''}
          onChange={(e) => setFilters({
            ...filters,
            dateRange: { ...filters.dateRange, end: e.target.value }
          })}
          InputLabelProps={{ shrink: true }}
        />
      </Box>

      <Divider sx={{ my: 2 }} />

      {/* Clear Filters Button */}
      <Button
        fullWidth
        variant="outlined"
        onClick={handleClearFilters}
        sx={{ mt: 2 }}
      >
        Clear All Filters
      </Button>
    </Paper>
  )
}

export default FilterPanel

