import { Box, Skeleton, Card, CardContent, Grid, Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material'

export function CardSkeleton({ count = 4 }) {
  return (
    <Grid container spacing={2} sx={{ mb: 4 }}>
      {Array.from({ length: count }).map((_, index) => (
        <Grid item xs={12} sm={6} md={3} key={index}>
          <Card>
            <CardContent>
              <Skeleton variant="text" width="60%" height={20} sx={{ mb: 1 }} />
              <Skeleton variant="text" width="80%" height={40} />
            </CardContent>
          </Card>
        </Grid>
      ))}
    </Grid>
  )
}

export function TableSkeleton({ rows = 5, cols = 7 }) {
  return (
    <TableContainer component={Paper}>
      <Table>
        <TableHead>
          <TableRow>
            {Array.from({ length: cols }).map((_, index) => (
              <TableCell key={index}>
                <Skeleton variant="text" width="80%" />
              </TableCell>
            ))}
          </TableRow>
        </TableHead>
        <TableBody>
          {Array.from({ length: rows }).map((_, rowIndex) => (
            <TableRow key={rowIndex}>
              {Array.from({ length: cols }).map((_, colIndex) => (
                <TableCell key={colIndex}>
                  <Skeleton variant="text" />
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  )
}

export function ChartSkeleton({ height = 400 }) {
  return (
    <Box sx={{ width: '100%', height, position: 'relative' }}>
      <Skeleton variant="rectangular" width="100%" height={height} />
      <Box sx={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }}>
        <Skeleton variant="circular" width={40} height={40} />
      </Box>
    </Box>
  )
}

export function MapSkeleton() {
  return (
    <Box sx={{ width: '100%', height: '100%', position: 'relative', minHeight: 400 }}>
      <Skeleton variant="rectangular" width="100%" height="100%" />
    </Box>
  )
}

