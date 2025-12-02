// API service for Flask backend
import axios from 'axios'

// Get API base URL from environment variable or use default
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000/api'

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 second timeout
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`)
    return config
  },
  (error) => {
    console.error('API Request Error:', error)
    return Promise.reject(error)
  }
)

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    return response
  },
  (error) => {
    if (error.response) {
      // Server responded with error status
      console.error('API Error:', error.response.status, error.response.data)
      if (error.response.status === 429) {
        alert('Rate limit exceeded. Please try again later.')
      }
    } else if (error.request) {
      // Request made but no response
      console.error('API Network Error:', error.request)
      alert('Network error. Please check if the backend server is running.')
    } else {
      console.error('API Error:', error.message)
    }
    return Promise.reject(error)
  }
)

// Normalize borough names for API (UI names -> Database names)
// Based on migration script, database stores: 'The Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island', 'Bridges'
const normalizeBoroughForAPI = (borough) => {
  // Map UI names to database names
  const mapping = {
    'Bronx': 'The Bronx', // UI might use 'Bronx', but DB has 'The Bronx'
    'The Bronx': 'The Bronx', // Already correct
    'Bridges': 'Bridges', // Already correct
    // All other boroughs should match exactly
  }
  return mapping[borough] || borough
}

// Normalize borough names from API (Database names -> UI names)
const normalizeBoroughFromAPI = (borough) => {
  // Map database names to UI names (if needed)
  // Currently they match, but this allows for future changes
  return borough
}

// Locations API
export const getLocations = async (params = {}) => {
  try {
    const apiParams = {}
    
    // Handle boroughs - normalize names and handle arrays
    if (params['borough[]'] || params.boroughs) {
      const boroughs = params['borough[]'] || params.boroughs || []
      const normalizedBoroughs = boroughs.flatMap(b => {
        const normalized = normalizeBoroughForAPI(b)
        return Array.isArray(normalized) ? normalized : [normalized]
      })
      apiParams['borough[]'] = normalizedBoroughs
    }
    
    if (params['category[]'] || params.categories) {
      apiParams['category[]'] = params['category[]'] || params.categories || []
    }
    
    if (params.min_count !== undefined || params.minCount !== undefined) {
      apiParams.min_count = params.min_count || params.minCount
    }
    
    if (params.max_count !== undefined || params.maxCount !== undefined) {
      apiParams.max_count = params.max_count || params.maxCount
    }
    
    if (params.search) {
      apiParams.search = params.search
    }
    
    if (params.format) {
      apiParams.format = params.format
    }
    
    const response = await api.get('/locations', { params: apiParams })
    
    // If format is geojson, return as-is
    if (params.format === 'geojson') {
      return response.data
    }
    
    // Otherwise return JSON format
    return response.data
  } catch (error) {
    console.error('Error fetching locations:', error)
    // Fallback to empty data on error
    return params.format === 'geojson' 
      ? { type: 'FeatureCollection', features: [] }
      : { locations: [], count: 0 }
  }
}

export const getLocation = async (id) => {
  try {
    const response = await api.get(`/locations/${id}`)
    return response.data
  } catch (error) {
    console.error('Error fetching location:', error)
    throw error
  }
}

export const getLocationsInBounds = async (bounds, params = {}) => {
  try {
    const apiParams = { bounds, ...params }
    
    // Normalize boroughs if present
    if (params['borough[]'] || params.boroughs) {
      const boroughs = params['borough[]'] || params.boroughs || []
      const normalizedBoroughs = boroughs.flatMap(b => {
        const normalized = normalizeBoroughForAPI(b)
        return Array.isArray(normalized) ? normalized : [normalized]
      })
      apiParams['borough[]'] = normalizedBoroughs
    }
    
    const response = await api.get('/locations/bounds', { params: apiParams })
    return response.data
  } catch (error) {
    console.error('Error fetching locations in bounds:', error)
    return { type: 'FeatureCollection', features: [] }
  }
}

// Statistics API
export const getSummaryStats = async (params = {}) => {
  try {
    const apiParams = {}
    
    if (params['borough[]'] || params.boroughs) {
      const boroughs = params['borough[]'] || params.boroughs || []
      const normalizedBoroughs = boroughs.flatMap(b => {
        const normalized = normalizeBoroughForAPI(b)
        return Array.isArray(normalized) ? normalized : [normalized]
      })
      apiParams['borough[]'] = normalizedBoroughs
    }
    
    if (params['category[]'] || params.categories) {
      apiParams['category[]'] = params['category[]'] || params.categories || []
    }
    
    if (params.min_count !== undefined || params.minCount !== undefined) {
      apiParams.min_count = params.min_count || params.minCount
    }
    
    if (params.max_count !== undefined || params.maxCount !== undefined) {
      apiParams.max_count = params.max_count || params.maxCount
    }
    
    const response = await api.get('/statistics/summary', { params: apiParams })
    return response.data
  } catch (error) {
    console.error('Error fetching summary stats:', error)
    return {
      total_locations: 0,
      mean_count: 0,
      median_count: 0,
      min_count: 0,
      max_count: 0,
      std_dev: 0
    }
  }
}

export const getBoroughStats = async () => {
  try {
    const response = await api.get('/statistics/by-borough')
    // Backend returns { statistics: [...] }, ensure it matches expected format
    return response.data
  } catch (error) {
    console.error('Error fetching borough stats:', error)
    return { statistics: [] }
  }
}

export const getCategoryStats = async () => {
  try {
    const response = await api.get('/statistics/by-category')
    // Backend returns { statistics: [...] }, ensure it matches expected format
    return response.data
  } catch (error) {
    console.error('Error fetching category stats:', error)
    return { statistics: [] }
  }
}

export const getTopSites = async (limit = 10) => {
  try {
    const response = await api.get('/statistics/top-sites', {
      params: { limit }
    })
    return response.data
  } catch (error) {
    console.error('Error fetching top sites:', error)
    return { sites: [], count: 0 }
  }
}

// Time Series API
export const getTimeSeries = async (locationId, params = {}) => {
  try {
    const apiParams = {}
    
    if (params.start_date) apiParams.start_date = params.start_date
    if (params.end_date) apiParams.end_date = params.end_date
    if (params['period[]'] || params.periods) {
      apiParams['period[]'] = params['period[]'] || params.periods || []
    }
    
    const response = await api.get(`/counts/time-series/${locationId}`, {
      params: apiParams
    })
    return response.data
  } catch (error) {
    console.error('Error fetching time series:', error)
    return { location_id: locationId, counts: [], total_records: 0 }
  }
}

export const getAggregateTimeSeries = async (params = {}) => {
  try {
    const response = await api.get('/counts/time-series/aggregate', { params })
    return response.data
  } catch (error) {
    console.error('Error fetching aggregate time series:', error)
    return { aggregated_data: [], total_records: 0 }
  }
}

// Comparison API
export const compareGroups = async (group1, group2) => {
  try {
    // Normalize borough names in groups
    const normalizeGroup = (group) => {
      if (group.type === 'borough' && group.values) {
        const normalizedValues = group.values.flatMap(v => {
          const normalized = normalizeBoroughForAPI(v)
          return Array.isArray(normalized) ? normalized : [normalized]
        })
        return { ...group, values: normalizedValues }
      }
      return group
    }
    
    const normalizedGroup1 = normalizeGroup(group1)
    const normalizedGroup2 = normalizeGroup(group2)
    
    const response = await api.post('/comparison', {
      group1: normalizedGroup1,
      group2: normalizedGroup2
    })
    return response.data
  } catch (error) {
    console.error('Error comparing groups:', error)
    throw error
  }
}

// Export API
export const exportCSV = async (params = {}) => {
  try {
    const apiParams = {}
    
    if (params['borough[]'] || params.boroughs) {
      const boroughs = params['borough[]'] || params.boroughs || []
      const normalizedBoroughs = boroughs.flatMap(b => {
        const normalized = normalizeBoroughForAPI(b)
        return Array.isArray(normalized) ? normalized : [normalized]
      })
      apiParams['borough[]'] = normalizedBoroughs
    }
    
    if (params['category[]'] || params.categories) {
      apiParams['category[]'] = params['category[]'] || params.categories || []
    }
    
    if (params.min_count !== undefined || params.minCount !== undefined) {
      apiParams.min_count = params.min_count || params.minCount
    }
    
    if (params.max_count !== undefined || params.maxCount !== undefined) {
      apiParams.max_count = params.max_count || params.maxCount
    }
    
    if (params.search) {
      apiParams.search = params.search
    }
    
    const response = await api.get('/export/csv', {
      params: apiParams,
      responseType: 'blob'
    })
    
    // Create download link
    const url = window.URL.createObjectURL(new Blob([response.data]))
    const link = document.createElement('a')
    link.href = url
    link.setAttribute('download', 'pedestrian_data.csv')
    document.body.appendChild(link)
    link.click()
    link.remove()
    window.URL.revokeObjectURL(url)
    
    return response.data
  } catch (error) {
    console.error('Error exporting CSV:', error)
    throw error
  }
}

export const exportGeoJSON = async (params = {}) => {
  try {
    const apiParams = {}
    
    if (params['borough[]'] || params.boroughs) {
      const boroughs = params['borough[]'] || params.boroughs || []
      const normalizedBoroughs = boroughs.flatMap(b => {
        const normalized = normalizeBoroughForAPI(b)
        return Array.isArray(normalized) ? normalized : [normalized]
      })
      apiParams['borough[]'] = normalizedBoroughs
    }
    
    if (params['category[]'] || params.categories) {
      apiParams['category[]'] = params['category[]'] || params.categories || []
    }
    
    if (params.min_count !== undefined || params.minCount !== undefined) {
      apiParams.min_count = params.min_count || params.minCount
    }
    
    if (params.max_count !== undefined || params.maxCount !== undefined) {
      apiParams.max_count = params.max_count || params.maxCount
    }
    
    if (params.search) {
      apiParams.search = params.search
    }
    
    const response = await api.get('/export/geojson', {
      params: apiParams,
      responseType: 'blob'
    })
    
    // Create download link
    const url = window.URL.createObjectURL(new Blob([response.data]))
    const link = document.createElement('a')
    link.href = url
    link.setAttribute('download', 'pedestrian_data.geojson')
    document.body.appendChild(link)
    link.click()
    link.remove()
    window.URL.revokeObjectURL(url)
    
    return response.data
  } catch (error) {
    console.error('Error exporting GeoJSON:', error)
    throw error
  }
}

// Health check
export const healthCheck = async () => {
  try {
    const response = await api.get('/health')
    return response.data
  } catch (error) {
    console.error('Health check failed:', error)
    throw new Error('Backend API not available')
  }
}

