# Step-by-Step Deployment Guide

Follow these steps to deploy your Alzheimer's Prediction System to production.

---

## 🎯 Prerequisites Checklist

Before starting, ensure you have:
- [ ] GitHub account (for easy deployment)
- [ ] Render.com account (free tier)
- [ ] Netlify account (free tier) OR Vercel account (free tier)
- [ ] Your `backend/models/best_model.pkl` file committed to Git

---

## 📦 Part 1: Deploy Backend to Render.com

### Step 1: Prepare Backend Repository

1. **Navigate to your backend directory:**
   ```bash
   cd backend
   ```

2. **Initialize Git (if not already done):**
   ```bash
   git init
   git add .
   git commit -m "Backend ready for deployment"
   ```

3. **Create GitHub repository:**
   - Go to [github.com](https://github.com)
   - Click "New repository"
   - Name it: `alzheimers-backend` (or your choice)
   - **DO NOT** initialize with README
   - Click "Create repository"

4. **Push to GitHub:**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/alzheimers-backend.git
   git branch -M main
   git push -u origin main
   ```
   *(Replace `YOUR_USERNAME` with your GitHub username)*

### Step 2: Deploy to Render

1. **Go to [Render.com](https://render.com)** and sign in (or sign up with GitHub)

2. **Create New Web Service:**
   - Click "New +" button (top right)
   - Select "Web Service"

3. **Connect Repository:**
   - Click "Connect GitHub" or "Connect GitLab"
   - Authorize Render
   - Select your `alzheimers-backend` repository

4. **Configure Service:**
   - **Name**: `alzheimers-api` (or your choice)
   - **Region**: Choose closest to you (Oregon, Frankfurt, etc.)
   - **Branch**: `main`
   - **Root Directory**: `backend` (if repo root) or leave blank if `backend/` is the repo root
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn src.api:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free

5. **Add Environment Variables:**
   Click "Advanced" → "Add Environment Variable"
   
   Add:
   ```
   ENVIRONMENT=production
   ```
   
   (Leave `FRONTEND_URL` empty for now - we'll add it after deploying frontend)

6. **Create Web Service:**
   - Click "Create Web Service"
   - Wait for deployment (5-10 minutes first time)

7. **Note Your Backend URL:**
   - Once deployed, your backend URL will be: `https://alzheimers-api.onrender.com`
   - (Or `https://YOUR-SERVICE-NAME.onrender.com`)
   - **Copy this URL** - you'll need it for frontend!

8. **Test Backend:**
   - Visit: `https://YOUR-BACKEND-URL.onrender.com/`
   - You should see: `{"message": "Alzheimer Risk Prediction API is running."}`

---

## 🌐 Part 2: Deploy Frontend to Netlify

### Step 1: Prepare Frontend

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Test build locally (optional but recommended):**
   ```bash
   npm install
   npm run build
   ```
   
   This creates a `dist/` folder. If this works, you're ready!

3. **Initialize Git (if not already done):**
   ```bash
   git init
   git add .
   git commit -m "Frontend ready for deployment"
   ```

4. **Create GitHub repository:**
   - Go to [github.com](https://github.com)
   - Create new repo: `alzheimers-frontend`
   - **DO NOT** initialize with README

5. **Push to GitHub:**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/alzheimers-frontend.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Deploy to Netlify

1. **Go to [Netlify](https://netlify.com)** and sign in (or sign up with GitHub)

2. **Add New Site:**
   - Click "Add new site"
   - Select "Import an existing project"

3. **Connect to Git:**
   - Choose "GitHub" (or GitLab/Bitbucket)
   - Authorize Netlify
   - Select your `alzheimers-frontend` repository

4. **Configure Build Settings:**
   - **Base directory**: `frontend` (if repo root is project root)
   - **Build command**: `npm run build`
   - **Publish directory**: `frontend/dist`
   - (If `frontend/` is your repo root, use: Build: `npm run build`, Publish: `dist`)

5. **Set Environment Variable:**
   - Click "Show advanced"
   - Click "New variable"
   - **Key**: `VITE_API_URL`
   - **Value**: `https://YOUR-BACKEND-URL.onrender.com`
   - (Replace with your actual Render backend URL from Part 1, Step 7)
   - Click "Add variable"

6. **Deploy:**
   - Click "Deploy site"
   - Wait for build (2-5 minutes)

7. **Note Your Frontend URL:**
   - Netlify will assign: `https://random-name-123.netlify.app`
   - Or set a custom site name: Site settings → Change site name
   - **Copy this URL** - you'll need it for backend CORS!

---

## 🔗 Part 3: Connect Backend and Frontend

### Step 1: Update Backend CORS

1. **Go back to Render Dashboard:**
   - Navigate to your backend service

2. **Add Frontend URL:**
   - Go to "Environment" tab
   - Click "Add Environment Variable"
   - **Key**: `FRONTEND_URL`
   - **Value**: `https://YOUR-FRONTEND-URL.netlify.app`
   - (Replace with your actual Netlify URL)
   - Click "Save Changes"

3. **Redeploy:**
   - Render will auto-redeploy, or click "Manual Deploy" → "Deploy latest commit"
   - Wait 2-3 minutes

### Step 2: Verify Everything Works

1. **Visit your frontend URL:**
   - Go to: `https://YOUR-FRONTEND-URL.netlify.app`

2. **Test the application:**
   - Fill in the form
   - Click "Analyze"
   - You should see predictions! 🎉

3. **Check for errors:**
   - Open browser DevTools (F12)
   - Go to "Console" tab
   - Look for any red errors
   - If you see CORS errors, double-check `FRONTEND_URL` in Render

---

## 🚀 Alternative: Deploy Frontend to Vercel

If you prefer Vercel over Netlify:

### Step 1: Install Vercel CLI
```bash
npm install -g vercel
```

### Step 2: Deploy
```bash
cd frontend
vercel
```

Follow prompts:
- **Set up and deploy?** → Yes
- **Which scope?** → Your account
- **Link to existing project?** → No
- **Project name?** → `alzheimers-frontend`
- **Directory?** → `./`
- **Override settings?** → No

### Step 3: Set Environment Variable

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Select your project
3. Go to **Settings** → **Environment Variables**
4. Add:
   - **Name**: `VITE_API_URL`
   - **Value**: `https://YOUR-BACKEND-URL.onrender.com`
   - **Environment**: Production, Preview, Development (check all)
5. Click "Save"
6. **Redeploy** (or it auto-redeploys)

### Step 4: Update Backend CORS

Same as Netlify - set `FRONTEND_URL` in Render to your Vercel URL.

---

## ✅ Final Checklist

- [ ] Backend deployed to Render
- [ ] Backend URL accessible (shows JSON message)
- [ ] Frontend deployed to Netlify/Vercel
- [ ] `VITE_API_URL` set in Netlify/Vercel
- [ ] `FRONTEND_URL` set in Render
- [ ] Backend redeployed after setting `FRONTEND_URL`
- [ ] Frontend can make predictions
- [ ] No console errors

---

## 🐛 Troubleshooting

### Backend Issues

**Problem: Model not found**
- **Solution**: Make sure `models/best_model.pkl` is committed to Git and pushed

**Problem: Build fails**
- **Solution**: Check Render logs → Check if `requirements.txt` is correct

**Problem: CORS errors**
- **Solution**: 
  1. Verify `FRONTEND_URL` is set correctly in Render
  2. Make sure URL starts with `https://`
  3. Redeploy backend

### Frontend Issues

**Problem: "Failed to get prediction"**
- **Solution**: 
  1. Check `VITE_API_URL` in Netlify/Vercel dashboard
  2. Verify backend URL is correct
  3. Check browser console for errors

**Problem: Build fails**
- **Solution**: 
  1. Test build locally: `npm run build`
  2. Fix any TypeScript/compilation errors
  3. Ensure all dependencies in `package.json`

**Problem: Blank page after deploy**
- **Solution**: 
  1. Check Netlify/Vercel build logs
  2. Verify `dist/` folder is being published
  3. Check browser console for runtime errors

---

## 🎉 Success!

Your Alzheimer's Risk Prediction System is now live on the internet!

**Your URLs:**
- Backend: `https://YOUR-BACKEND-URL.onrender.com`
- Frontend: `https://YOUR-FRONTEND-URL.netlify.app`

Share your frontend URL with others to test your application!

---

## 📝 Quick Reference

### Environment Variables Summary

**Render (Backend):**
- `ENVIRONMENT=production`
- `FRONTEND_URL=https://your-frontend.netlify.app`

**Netlify/Vercel (Frontend):**
- `VITE_API_URL=https://your-backend.onrender.com`

---

## 🔄 Auto-Deployments

Both Render and Netlify/Vercel support automatic deployments:
- **Push to `main` branch** → Automatic deployment
- No manual steps needed after initial setup!

---

Need help? Check the logs in:
- Render: Service → Logs
- Netlify: Site → Deploys → Click deploy → Build logs
- Vercel: Deployments → Click deployment → Build logs

