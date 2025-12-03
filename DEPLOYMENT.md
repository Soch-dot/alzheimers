# Deployment Guide - Alzheimer's Risk Prediction System

This guide walks you through deploying both the **backend (FastAPI)** and **frontend (React)** for free using Render.com and Netlify/Vercel.

---

## 📋 Prerequisites

- GitHub account (recommended for easy deployment)
- Render.com account (free tier for backend)
- Netlify or Vercel account (free tier for frontend)
- Your trained model file: `backend/models/best_model.pkl`

---

## 🚀 Step 1: Deploy Backend to Render.com

### 1.1 Prepare Your Backend

Ensure these files exist in your `backend/` directory:
- ✅ `Procfile` - Tells Render how to run your app
- ✅ `runtime.txt` - Specifies Python version
- ✅ `requirements.txt` - Lists dependencies
- ✅ `src/api.py` - Your FastAPI app
- ✅ `models/best_model.pkl` - Your trained model

### 1.2 Push to GitHub (Optional but Recommended)

```bash
# Initialize git if not already done
cd backend
git init
git add .
git commit -m "Initial commit"

# Create a GitHub repo and push
git remote add origin <your-github-repo-url>
git push -u origin main
```

### 1.3 Deploy to Render

1. **Go to [Render.com](https://render.com)** and sign up/login
2. **Click "New +" → "Web Service"**
3. **Connect your GitHub repository** (or use "Public Git repository" and paste your repo URL)
4. **Configure the service:**
   - **Name**: `alzheimers-api` (or your choice)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn src.api:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free

5. **Set Environment Variables** (in Render dashboard):
   - `ENVIRONMENT` = `production`
   - `FRONTEND_URL` = Leave empty for now (will set after frontend deploys)

6. **Click "Create Web Service"**

7. **Wait for deployment** (first build takes 5-10 minutes)

8. **Note your backend URL**: It will be something like `https://alzheimers-api.onrender.com`

### 1.4 Verify Backend is Running

Visit `https://your-backend-url.onrender.com/` - you should see:
```json
{"message": "Alzheimer Risk Prediction API is running."}
```

---

## 🌐 Step 2: Deploy Frontend to Netlify

### 2.1 Update Frontend API URL

Before deploying, you need to set the backend URL. You can either:

**Option A: Use Environment Variable (Recommended)**

1. In your frontend directory, create `.env.production`:
```bash
VITE_API_URL=https://your-backend-name.onrender.com
```
*(Replace with your actual Render backend URL)*

2. **OR** set it in Netlify dashboard (see step 2.3)

### 2.2 Build Locally (Test First - Optional)

```bash
cd frontend
npm install
npm run build
```

This creates a `dist/` folder with production files.

### 2.3 Deploy to Netlify

**Method 1: Deploy via Netlify Dashboard**

1. **Go to [Netlify](https://netlify.com)** and sign up/login
2. **Drag and drop** your `frontend/dist` folder onto Netlify dashboard
3. **OR** click "Add new site" → "Import an existing project"
4. **Connect to GitHub** (if you pushed to GitHub)
5. **Configure build settings:**
   - **Base directory**: `frontend`
   - **Build command**: `npm run build`
   - **Publish directory**: `frontend/dist`

6. **Set Environment Variables** (Site settings → Environment variables):
   - `VITE_API_URL` = `https://your-backend-name.onrender.com`
   *(Replace with your actual Render backend URL)*

7. **Deploy!**

**Method 2: Deploy via Netlify CLI**

```bash
# Install Netlify CLI
npm install -g netlify-cli

# Login
netlify login

# Deploy
cd frontend
netlify deploy --prod
```

### 2.4 Update Backend CORS (Important!)

After deploying frontend, update your Render backend:

1. Go to Render dashboard → Your backend service → Environment
2. Set `FRONTEND_URL` = `https://your-netlify-site.netlify.app`
3. **Redeploy** the backend (or it will auto-redeploy)

This allows your frontend to communicate with the backend.

---

## 🚀 Alternative: Deploy Frontend to Vercel

### 2.1 Install Vercel CLI

```bash
npm install -g vercel
```

### 2.2 Deploy

```bash
cd frontend
vercel
```

Follow the prompts:
- **Link to existing project?** → No (first time)
- **Project name?** → `alzheimers-frontend` (or your choice)
- **Directory?** → `./`
- **Override settings?** → No

### 2.3 Set Environment Variables

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Select your project
3. Go to **Settings → Environment Variables**
4. Add:
   - **Name**: `VITE_API_URL`
   - **Value**: `https://your-backend-name.onrender.com`
   - **Environment**: Production, Preview, Development

5. **Redeploy** (or it auto-redeploys)

### 2.4 Update Backend CORS

Same as Netlify - set `FRONTEND_URL` in Render to your Vercel URL.

---

## ✅ Step 3: Verify Everything Works

1. **Visit your frontend URL** (Netlify or Vercel)
2. **Fill in the form** with sample data
3. **Click "Analyze"**
4. **You should see predictions!** 🎉

---

## 🔧 Troubleshooting

### Backend Issues

**Problem**: Model file not found
- **Solution**: Ensure `models/best_model.pkl` is in your Git repository
- Render needs access to the model file

**Problem**: CORS errors in browser
- **Solution**: 
  1. Check `FRONTEND_URL` is set in Render dashboard
  2. Ensure it matches your frontend URL exactly (with `https://`)
  3. Redeploy backend after setting the variable

**Problem**: Backend times out
- **Solution**: Render free tier spins down after 15 min inactivity. First request after spin-down takes longer. Consider upgrading to paid tier for always-on.

### Frontend Issues

**Problem**: "Failed to get prediction" error
- **Solution**: 
  1. Check `VITE_API_URL` is set correctly in Netlify/Vercel
  2. Verify backend URL is accessible (visit it in browser)
  3. Check browser console for CORS errors

**Problem**: Build fails
- **Solution**: 
  1. Test build locally: `npm run build`
  2. Fix any TypeScript/compilation errors
  3. Ensure all dependencies are in `package.json`

---

## 📝 Environment Variables Summary

### Backend (Render.com)

| Variable | Value | Required |
|----------|-------|----------|
| `ENVIRONMENT` | `production` | Optional |
| `FRONTEND_URL` | `https://your-frontend.netlify.app` | **Yes** (after frontend deploys) |
| `MODEL_PATH` | Custom path (if needed) | Optional |

### Frontend (Netlify/Vercel)

| Variable | Value | Required |
|----------|-------|----------|
| `VITE_API_URL` | `https://your-backend.onrender.com` | **Yes** |

---

## 💰 Cost

- **Render.com (Backend)**: Free tier (with limitations)
- **Netlify (Frontend)**: Free tier (generous limits)
- **Vercel (Frontend)**: Free tier (generous limits)

**Total Cost: $0/month** ✨

---

## 🔄 Continuous Deployment

Both Render and Netlify/Vercel support auto-deployment:
- Push to your `main` branch → Automatic deployment
- No manual steps needed after initial setup!

---

## 📚 Additional Resources

- [Render Documentation](https://render.com/docs)
- [Netlify Documentation](https://docs.netlify.com)
- [Vercel Documentation](https://vercel.com/docs)

---

## 🎉 You're Done!

Your Alzheimer's Risk Prediction system is now live and accessible worldwide!

**Next Steps:**
- Share your deployed frontend URL with others
- Monitor usage in Render/Netlify dashboards
- Consider adding custom domain names (free on Netlify/Vercel)

