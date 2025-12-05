/**
 * CSV Data Service - Loads data from CSV files instead of API
 */

let cachedData = null
let cachedGeoJSON = null

// Load CSV data
export const loadCSVData = async () => {
  if (cachedData) {
    return cachedData
  }

  try {
    const response = await fetch('/pedestrian_data.csv')
    const text = await response.text()
    const lines = text.split('\n').filter(line => line.trim())
    const headers = lines[0].split(',')

    const data = lines.slice(1).map(line => {
      const values = line.split(',')
      const obj = {}
      headers.forEach((header, index) => {
        let value = values[index] || ''
        value = value.trim().replace(/^"|"$/g, '') // Remove quotes
        
        // Parse numeric values
        if (header === 'avg_recent_count' || header === 'segmentid' || header === 'Loc' || header === 'OBJECTID') {
          value = value ? parseFloat(value) : null
        }
        
        obj[header] = value
      })
      return obj
    }).filter(row => row.OBJECTID) // Filter out empty rows

    cachedData = data
    return data
  } catch (error) {
    console.error('Error loading CSV data:', error)
    throw error
  }
}

// Load GeoJSON data
export const loadGeoJSONData = async () => {
  if (cachedGeoJSON) {
    return cachedGeoJSON
  }

  try {
    const response = await fetch('/pedestrian_data.geojson')
    const geojson = await response.json()
    cachedGeoJSON = geojson
    return geojson
  } catch (error) {
    console.warn('GeoJSON not available, using CSV coordinates:', error)
    // Fallback: create GeoJSON from CSV if available
    const csvData = await loadCSVData()
    return csvToGeoJSON(csvData)
  }
}

// Convert CSV data to GeoJSON format
const csvToGeoJSON = (csvData) => {
  const features = csvData
    .filter(row => row.latitude && row.longitude)
    .map(row => ({
      type: 'Feature',
      geometry: {
        type: 'Point',
        coordinates: [parseFloat(row.longitude), parseFloat(row.latitude)]
      },
      properties: {
        id: row.OBJECTID,
        objectid: row.OBJECTID,
        loc_id: row.Loc,
        borough: row.Borough,
        street_name_clean: row.Street_Nam_clean,
        street_clean: row.street_clean,
        category: row.Category,
        segmentid: row.segmentid,
        avg_recent_count: row.avg_recent_count
      }
    }))

  return {
    type: 'FeatureCollection',
    features
  }
}

// Get locations with filters
export const getLocations = async (filters = {}) => {
  const csvData = await loadCSVData()
  
  let filtered = [...csvData]

  // Normalize borough names - map UI names to CSV names
  const normalizeBoroughName = (uiName, uniqueBoroughs) => {
    // Check if the UI name matches exactly
    if (uniqueBoroughs.includes(uiName)) {
      return [uiName]
    }
    
    // Try case-insensitive match
    const caseInsensitiveMatch = uniqueBoroughs.find(b => 
      b && b.toLowerCase() === uiName.toLowerCase()
    )
    if (caseInsensitiveMatch) {
      return [caseInsensitiveMatch]
    }
    
    // Try partial match (e.g., "The Bronx" vs "Bronx")
    const partialMatch = uniqueBoroughs.find(b => 
      b && (b.includes(uiName) || uiName.includes(b))
    )
    if (partialMatch) {
      return [partialMatch]
    }
    
    // Special mappings
    const boroughMap = {
      'The Bronx': ['Bronx', 'The Bronx'],
      'Bronx': ['Bronx', 'The Bronx'],
      'Bridges': ['Bridges', 'East River Bridges', 'Harlem River Bridges']
    }
    
    if (boroughMap[uiName]) {
      return boroughMap[uiName].filter(b => uniqueBoroughs.includes(b))
    }
    
    return [uiName] // Return as-is if no mapping found
  }

  // Apply filters
  if (filters.boroughs && filters.boroughs.length > 0) {
    const uniqueBoroughs = [...new Set(csvData.map(row => row.Borough).filter(Boolean))]
    const normalizedBoroughs = filters.boroughs.flatMap(name => normalizeBoroughName(name, uniqueBoroughs))
    filtered = filtered.filter(row => normalizedBoroughs.includes(row.Borough))
  }

  if (filters.categories && filters.categories.length > 0) {
    filtered = filtered.filter(row => filters.categories.includes(row.Category))
  }

  if (filters.minCount !== null && filters.minCount !== undefined) {
    filtered = filtered.filter(row => row.avg_recent_count >= filters.minCount)
  }

  if (filters.maxCount !== null && filters.maxCount !== undefined) {
    filtered = filtered.filter(row => row.avg_recent_count <= filters.maxCount)
  }

  if (filters.search) {
    const searchLower = filters.search.toLowerCase()
    filtered = filtered.filter(row => 
      (row.Street_Nam_clean && row.Street_Nam_clean.toLowerCase().includes(searchLower)) ||
      (row.street_clean && row.street_clean.toLowerCase().includes(searchLower)) ||
      (row.Loc && row.Loc.toString().includes(searchLower))
    )
  }

  // Convert to GeoJSON format
  const geojson = await loadGeoJSONData()
  const geojsonMap = new Map(
    geojson.features.map(f => [parseInt(f.properties.OBJECTID || f.properties.objectid), f])
  )

  const features = filtered
    .map(row => {
      const objectId = parseInt(row.OBJECTID)
      const feature = geojsonMap.get(objectId)
      if (feature) {
        return {
          ...feature,
          properties: {
            ...feature.properties,
            objectid: objectId,
            id: objectId,
            loc_id: row.Loc,
            borough: row.Borough,
            street_name_clean: row.Street_Nam_clean || feature.properties.Street_Nam_clean,
            street_clean: row.street_clean || feature.properties.street_clean,
            category: row.Category,
            segmentid: row.segmentid,
            avg_recent_count: row.avg_recent_count
          }
        }
      }
      // Fallback: create feature from CSV if no GeoJSON
      return null
    })
    .filter(f => f !== null && f.geometry && f.geometry.coordinates && f.geometry.coordinates.length === 2)

  return {
    type: 'FeatureCollection',
    features
  }
}

// Get summary statistics
export const getSummaryStats = async (filters = {}) => {
  const csvData = await loadCSVData()
  
  let filtered = [...csvData]

  // Apply filters
  if (filters.boroughs && filters.boroughs.length > 0) {
    filtered = filtered.filter(row => filters.boroughs.includes(row.Borough))
  }
  if (filters.categories && filters.categories.length > 0) {
    filtered = filtered.filter(row => filters.categories.includes(row.Category))
  }
  if (filters.minCount !== null && filters.minCount !== undefined) {
    filtered = filtered.filter(row => row.avg_recent_count >= filters.minCount)
  }
  if (filters.maxCount !== null && filters.maxCount !== undefined) {
    filtered = filtered.filter(row => row.avg_recent_count <= filters.maxCount)
  }

  const counts = filtered.map(row => row.avg_recent_count).filter(c => c !== null && !isNaN(c))
  
  if (counts.length === 0) {
    return {
      total_locations: 0,
      mean_count: 0,
      median_count: 0,
      min_count: 0,
      max_count: 0,
      std_dev: 0
    }
  }

  const mean = counts.reduce((a, b) => a + b, 0) / counts.length
  const sorted = [...counts].sort((a, b) => a - b)
  const median = sorted.length % 2 === 0
    ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
    : sorted[Math.floor(sorted.length / 2)]
  const min = Math.min(...counts)
  const max = Math.max(...counts)
  const variance = counts.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / counts.length
  const stdDev = Math.sqrt(variance)

  return {
    total_locations: filtered.length,
    mean_count: mean,
    median_count: median,
    min_count: min,
    max_count: max,
    std_dev: stdDev
  }
}

// Get statistics by borough
export const getBoroughStats = async () => {
  const csvData = await loadCSVData()
  
  const boroughMap = new Map()
  
  csvData.forEach(row => {
    if (!row.Borough || !row.avg_recent_count) return
    
    if (!boroughMap.has(row.Borough)) {
      boroughMap.set(row.Borough, [])
    }
    boroughMap.get(row.Borough).push(row.avg_recent_count)
  })

  const statistics = Array.from(boroughMap.entries()).map(([borough, counts]) => {
    const mean = counts.reduce((a, b) => a + b, 0) / counts.length
    const sorted = [...counts].sort((a, b) => a - b)
    const median = sorted.length % 2 === 0
      ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
      : sorted[Math.floor(sorted.length / 2)]
    const min = Math.min(...counts)
    const max = Math.max(...counts)
    const variance = counts.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / counts.length
    const stdDev = Math.sqrt(variance)

    return {
      borough,
      location_count: counts.length,
      avg_count: mean,
      median_count: median,
      min_count: min,
      max_count: max,
      std_dev: stdDev
    }
  })

  return { statistics }
}

// Get statistics by category
export const getCategoryStats = async () => {
  const csvData = await loadCSVData()
  
  const categoryMap = new Map()
  
  csvData.forEach(row => {
    if (!row.Category || !row.avg_recent_count) return
    
    if (!categoryMap.has(row.Category)) {
      categoryMap.set(row.Category, [])
    }
    categoryMap.get(row.Category).push(row.avg_recent_count)
  })

  const statistics = Array.from(categoryMap.entries()).map(([category, counts]) => {
    const mean = counts.reduce((a, b) => a + b, 0) / counts.length
    const sorted = [...counts].sort((a, b) => a - b)
    const median = sorted.length % 2 === 0
      ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
      : sorted[Math.floor(sorted.length / 2)]
    const min = Math.min(...counts)
    const max = Math.max(...counts)
    const variance = counts.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / counts.length
    const stdDev = Math.sqrt(variance)

    return {
      category,
      location_count: counts.length,
      avg_count: mean,
      median_count: median,
      min_count: min,
      max_count: max,
      std_dev: stdDev
    }
  })

  // Sort by avg_count in descending order (highest first)
  statistics.sort((a, b) => (b.avg_count || 0) - (a.avg_count || 0))

  return { statistics }
}

// Get top sites
export const getTopSites = async (limit = 10, borough = null) => {
  const csvData = await loadCSVData()
  
  let filtered = [...csvData]
    .filter(row => row.avg_recent_count !== null && !isNaN(row.avg_recent_count))
  
  // Filter by borough if specified
  if (borough && borough !== '') {
    // Normalize borough name to handle variations
    const uniqueBoroughs = [...new Set(csvData.map(row => row.Borough).filter(Boolean))]
    const normalizedBoroughs = []
    
    // Check for exact match
    if (uniqueBoroughs.includes(borough)) {
      normalizedBoroughs.push(borough)
    } else {
      // Try case-insensitive match
      const caseInsensitiveMatch = uniqueBoroughs.find(b => 
        b && b.toLowerCase() === borough.toLowerCase()
      )
      if (caseInsensitiveMatch) {
        normalizedBoroughs.push(caseInsensitiveMatch)
      } else {
        // Try partial match (e.g., "The Bronx" vs "Bronx")
        const partialMatch = uniqueBoroughs.find(b => 
          b && (b.includes(borough) || borough.includes(b))
        )
        if (partialMatch) {
          normalizedBoroughs.push(partialMatch)
        } else {
          // Special mappings
          const boroughMap = {
            'The Bronx': ['Bronx', 'The Bronx'],
            'Bronx': ['Bronx', 'The Bronx'],
            'Bridges': ['Bridges', 'East River Bridges', 'Harlem River Bridges']
          }
          if (boroughMap[borough]) {
            normalizedBoroughs.push(...boroughMap[borough].filter(b => uniqueBoroughs.includes(b)))
          } else {
            normalizedBoroughs.push(borough) // Use as-is
          }
        }
      }
    }
    
    filtered = filtered.filter(row => normalizedBoroughs.includes(row.Borough))
  }
  
  const sorted = filtered
    .sort((a, b) => (b.avg_recent_count || 0) - (a.avg_recent_count || 0))
    .slice(0, limit)
    .map(row => ({
      location: {
        id: row.OBJECTID,
        objectid: row.OBJECTID,
        loc_id: row.Loc,
        borough: row.Borough,
        street_name_clean: row.Street_Nam_clean,
        street_clean: row.street_clean,
        category: row.Category,
        segmentid: row.segmentid
      },
      avg_recent_count: row.avg_recent_count
    }))

  return { sites: sorted, count: sorted.length }
}

// Get count of sites by borough (for max limit calculation)
export const getSiteCountByBorough = async (borough = null) => {
  const csvData = await loadCSVData()
  
  let filtered = [...csvData]
    .filter(row => row.avg_recent_count !== null && !isNaN(row.avg_recent_count))
  
  // Filter by borough if specified
  if (borough && borough !== '') {
    // Normalize borough name to handle variations
    const uniqueBoroughs = [...new Set(csvData.map(row => row.Borough).filter(Boolean))]
    const normalizedBoroughs = []
    
    // Check for exact match
    if (uniqueBoroughs.includes(borough)) {
      normalizedBoroughs.push(borough)
    } else {
      // Try case-insensitive match
      const caseInsensitiveMatch = uniqueBoroughs.find(b => 
        b && b.toLowerCase() === borough.toLowerCase()
      )
      if (caseInsensitiveMatch) {
        normalizedBoroughs.push(caseInsensitiveMatch)
      } else {
        // Try partial match (e.g., "The Bronx" vs "Bronx")
        const partialMatch = uniqueBoroughs.find(b => 
          b && (b.includes(borough) || borough.includes(b))
        )
        if (partialMatch) {
          normalizedBoroughs.push(partialMatch)
        } else {
          // Special mappings
          const boroughMap = {
            'The Bronx': ['Bronx', 'The Bronx'],
            'Bronx': ['Bronx', 'The Bronx'],
            'Bridges': ['Bridges', 'East River Bridges', 'Harlem River Bridges']
          }
          if (boroughMap[borough]) {
            normalizedBoroughs.push(...boroughMap[borough].filter(b => uniqueBoroughs.includes(b)))
          } else {
            normalizedBoroughs.push(borough) // Use as-is
          }
        }
      }
    }
    
    filtered = filtered.filter(row => normalizedBoroughs.includes(row.Borough))
  }
  
  return filtered.length
}

// Compare groups
export const compareGroups = async (group1, group2) => {
  const csvData = await loadCSVData()
  
  // Get all unique borough names from the data to see what's actually there
  const uniqueBoroughs = [...new Set(csvData.map(row => row.Borough).filter(Boolean))]
  console.log('Available boroughs in data:', uniqueBoroughs)
  
  // Normalize borough names - map UI names to CSV names
  const normalizeBoroughName = (uiName) => {
    // Check if the UI name matches exactly
    if (uniqueBoroughs.includes(uiName)) {
      return [uiName]
    }
    
    // Try case-insensitive match
    const caseInsensitiveMatch = uniqueBoroughs.find(b => 
      b && b.toLowerCase() === uiName.toLowerCase()
    )
    if (caseInsensitiveMatch) {
      return [caseInsensitiveMatch]
    }
    
    // Try partial match (e.g., "The Bronx" vs "Bronx")
    const partialMatch = uniqueBoroughs.find(b => 
      b && (b.includes(uiName) || uiName.includes(b))
    )
    if (partialMatch) {
      return [partialMatch]
    }
    
    // Special mappings
    const boroughMap = {
      'The Bronx': ['Bronx', 'The Bronx'],
      'Bronx': ['Bronx', 'The Bronx'],
      'Bridges': ['Bridges', 'East River Bridges', 'Harlem River Bridges']
    }
    
    if (boroughMap[uiName]) {
      return boroughMap[uiName].filter(b => uniqueBoroughs.includes(b))
    }
    
    return [uiName] // Return as-is if no mapping found
  }
  
  const getGroupStats = (group) => {
    let filtered = [...csvData]
    
    if (group.type === 'borough') {
      // Normalize borough names and check against all variants
      const normalizedValues = group.values.flatMap(name => normalizeBoroughName(name))
      console.log(`Filtering for ${group.type}:`, group.values, '-> normalized:', normalizedValues)
      filtered = filtered.filter(row => {
        const rowBorough = row.Borough || ''
        return normalizedValues.includes(rowBorough)
      })
      console.log(`Filtered ${filtered.length} rows for ${group.values.join(', ')}`)
    } else if (group.type === 'category') {
      filtered = filtered.filter(row => group.values.includes(row.Category))
    }
    
    const counts = filtered.map(row => row.avg_recent_count).filter(c => c !== null && !isNaN(c))
    
    if (counts.length === 0) {
      return { count: 0, mean: 0, median: 0, min: 0, max: 0, std_dev: 0 }
    }
    
    const mean = counts.reduce((a, b) => a + b, 0) / counts.length
    const sorted = [...counts].sort((a, b) => a - b)
    const median = sorted.length % 2 === 0
      ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
      : sorted[Math.floor(sorted.length / 2)]
    const min = Math.min(...counts)
    const max = Math.max(...counts)
    const variance = counts.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / counts.length
    const stdDev = Math.sqrt(variance)
    
    return {
      count: filtered.length,
      mean,
      median,
      min,
      max,
      std_dev: stdDev
    }
  }
  
  const stats1 = getGroupStats(group1)
  const stats2 = getGroupStats(group2)
  
  const countDiff = stats1.count - stats2.count
  const countDiffPct = stats2.count > 0 ? (countDiff / stats2.count) * 100 : 0
  
  const meanDiff = stats1.mean - stats2.mean
  const meanDiffPct = stats2.mean > 0 ? (meanDiff / stats2.mean) * 100 : 0
  
  const medianDiff = stats1.median - stats2.median
  const medianDiffPct = stats2.median > 0 ? (medianDiff / stats2.median) * 100 : 0
  
  const maxDiff = stats1.max - stats2.max
  const maxDiffPct = stats2.max > 0 ? (maxDiff / stats2.max) * 100 : 0
  
  return {
    group1: {
      type: group1.type,
      values: group1.values,
      statistics: stats1
    },
    group2: {
      type: group2.type,
      values: group2.values,
      statistics: stats2
    },
    differences: {
      count: {
        absolute: countDiff,
        percentage: Math.round(countDiffPct * 100) / 100
      },
      mean: {
        absolute: Math.round(meanDiff * 100) / 100,
        percentage: Math.round(meanDiffPct * 100) / 100
      },
      median: {
        absolute: Math.round(medianDiff * 100) / 100,
        percentage: Math.round(medianDiffPct * 100) / 100
      },
      max: {
        absolute: Math.round(maxDiff * 100) / 100,
        percentage: Math.round(maxDiffPct * 100) / 100
      }
    }
  }
}

// Export CSV
export const exportCSV = async (filters = {}) => {
  const csvData = await loadCSVData()
  
  let filtered = [...csvData]
  
  // Apply filters (same as getLocations)
  if (filters.boroughs && filters.boroughs.length > 0) {
    filtered = filtered.filter(row => filters.boroughs.includes(row.Borough))
  }
  if (filters.categories && filters.categories.length > 0) {
    filtered = filtered.filter(row => filters.categories.includes(row.Category))
  }
  if (filters.minCount !== null && filters.minCount !== undefined) {
    filtered = filtered.filter(row => row.avg_recent_count >= filters.minCount)
  }
  if (filters.maxCount !== null && filters.maxCount !== undefined) {
    filtered = filtered.filter(row => row.avg_recent_count <= filters.maxCount)
  }
  if (filters.search) {
    const searchLower = filters.search.toLowerCase()
    filtered = filtered.filter(row => 
      (row.Street_Nam_clean && row.Street_Nam_clean.toLowerCase().includes(searchLower)) ||
      (row.street_clean && row.street_clean.toLowerCase().includes(searchLower))
    )
  }
  
  // Convert to CSV string
  const headers = Object.keys(filtered[0] || {})
  const csvRows = [
    headers.join(','),
    ...filtered.map(row => headers.map(header => {
      const value = row[header] || ''
      return `"${value}"`
    }).join(','))
  ]
  
  return new Blob([csvRows.join('\n')], { type: 'text/csv' })
}

// Export GeoJSON
export const exportGeoJSON = async (filters = {}) => {
  const geojson = await getLocations(filters)
  return new Blob([JSON.stringify(geojson, null, 2)], { type: 'application/json' })
}

// Get time series data for a location
export const getTimeSeries = async (locationId) => {
  const geojson = await loadGeoJSONData()
  
  // Find the feature with matching OBJECTID
  const feature = geojson.features.find(f => 
    parseInt(f.properties.OBJECTID || f.properties.objectid) === parseInt(locationId)
  )
  
  if (!feature) {
    return { location_id: locationId, counts: [] }
  }
  
  const props = feature.properties
  const counts = []
  
  // Parse time series columns (format: May07_AM, Sept07_PM, etc.)
  const timeSeriesPattern = /^(May|Sept|Oct|June|Apr|Mar|Feb|Jan|Nov|Dec)(\d{2})_(AM|PM|MD)$/
  
  // Look for columns ending with _num (numeric versions)
  Object.keys(props).forEach(key => {
    if (key.endsWith('_num')) {
      const baseKey = key.replace('_num', '')
      const match = baseKey.match(timeSeriesPattern)
      
      if (match) {
        const [, monthStr, yearStr, period] = match
        const year = 2000 + parseInt(yearStr)
        
        const monthMap = {
          'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6,
          'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sept': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
        const month = monthMap[monthStr] || 1
        
        // Use 15th as approximate day
        const date = new Date(year, month - 1, 15)
        const countValue = parseFloat(props[key])
        
        if (!isNaN(countValue) && countValue > 0) {
          counts.push({
            count_date: date.toISOString().split('T')[0],
            period: period,
            count_value: Math.round(countValue) // Round to whole number
          })
        }
      }
    }
  })
  
  // Sort by date and period
  counts.sort((a, b) => {
    const dateCompare = a.count_date.localeCompare(b.count_date)
    if (dateCompare !== 0) return dateCompare
    const periodOrder = { 'AM': 1, 'MD': 2, 'PM': 3 }
    return (periodOrder[a.period] || 0) - (periodOrder[b.period] || 0)
  })
  
  return {
    location_id: locationId,
    location: {
      id: parseInt(props.OBJECTID || props.objectid),
      objectid: parseInt(props.OBJECTID || props.objectid),
      loc_id: props.Loc,
      borough: props.Borough,
      street_name_clean: props.Street_Nam_clean,
      street_clean: props.street_clean,
      category: props.Category
    },
    counts: counts,
    total_records: counts.length
  }
}

