// Configuration for different environments
const ENV_CONFIG = {
    development: {
        API_URL: "http://localhost:8000/api/age-check"
    },
    production: {
        // UPDATE THIS after deploying backend to Railway
        // Example: "https://your-backend-name.railway.app/api/age-check"
        API_URL: "REPLACE_WITH_RAILWAY_BACKEND_URL/api/age-check"
    }
};

// Auto-detect environment
const isProduction = window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1';
const environment = isProduction ? 'production' : 'development';

// Export configuration
export const config = ENV_CONFIG[environment];
