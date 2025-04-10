// API Configuration
export const API_CONFIG = {
  // Use proxy path for API requests (to avoid CORS)
  BASE_URL: '/api', 
  TOKEN: process.env.NEXT_PUBLIC_API_TOKEN || 'backendapi1234567890',
  DEFAULT_RECORDS: process.env.NEXT_PUBLIC_DEFAULT_RECORDS || '100'
}; 