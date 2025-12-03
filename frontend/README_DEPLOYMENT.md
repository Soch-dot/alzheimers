# Frontend Deployment Quick Reference

## Environment Variables

Create a `.env.production` file in the `frontend/` directory with:

```bash
VITE_API_URL=https://your-backend-name.onrender.com
```

Replace `your-backend-name.onrender.com` with your actual Render backend URL.

## For Netlify/Vercel

Set the environment variable in the dashboard:
- **Variable Name**: `VITE_API_URL`
- **Value**: Your Render backend URL (e.g., `https://alzheimers-api.onrender.com`)

## Local Development

For local development, create `.env.local`:
```bash
VITE_API_URL=http://127.0.0.1:8000
```

## Build Commands

```bash
# Install dependencies
npm install

# Development
npm run dev

# Production build
npm run build

# Preview production build
npm run preview
```

