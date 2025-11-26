# Alzheimer's Risk Prediction - Frontend

A clean, minimalistic React frontend for the Alzheimer's prediction system.

## Features

- **Professional Design**: Apple-inspired minimalistic UI with Tailwind CSS
- **Patient Data Form**: Easy-to-use form for entering clinical data
- **Real-time Predictions**: Connects to FastAPI backend for instant predictions
- **Visual Results**: Beautiful probability bars showing prediction confidence

## Getting Started

### Prerequisites

- Node.js and npm installed
- Backend API running on `http://127.0.0.1:8000`

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

The app will open at `http://localhost:5173` (or another port if 5173 is busy).

### Build for Production

```bash
npm run build
```

The built files will be in the `dist` folder, ready to deploy to GitHub Pages, Vercel, or Netlify.

## Project Structure

```
frontend/
├── src/
│   ├── App.tsx          # Main application component
│   ├── api.ts           # API service for backend communication
│   ├── main.tsx         # React entry point
│   └── style.css        # Tailwind CSS imports
├── index.html           # HTML template
├── tailwind.config.js  # Tailwind configuration
└── package.json         # Dependencies
```

## Usage

1. Make sure the backend API is running (see backend README)
2. Start the frontend: `npm run dev`
3. Fill in the patient clinical data form
4. Click "Get Prediction" to see the results
5. Use "Load Sample Data" to test with example values

## Deployment

### GitHub Pages

1. Build the project: `npm run build`
2. Push to GitHub
3. Enable GitHub Pages in repository settings
4. Set source to `dist` folder

### Vercel (Recommended)

1. Install Vercel CLI: `npm i -g vercel`
2. Run `vercel` in the frontend folder
3. Follow the prompts

### Netlify

1. Build the project: `npm run build`
2. Drag and drop the `dist` folder to Netlify
3. Or connect your GitHub repo for automatic deployments

## Notes

- The frontend expects the backend API at `http://127.0.0.1:8000`
- For production, update the API URL in `src/api.ts` to your deployed backend URL
- All styling uses Tailwind CSS for easy customization

