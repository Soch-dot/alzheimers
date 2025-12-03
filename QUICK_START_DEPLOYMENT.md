# 🚀 Quick Start - Deployment in 10 Minutes

## Prerequisites
- GitHub account
- Render.com account (free)
- Netlify account (free)

---

## Step 1: Deploy Backend (5 minutes)

1. **Push backend to GitHub:**
   ```bash
   cd backend
   git init
   git add .
   git commit -m "Ready for deployment"
   # Create repo on GitHub, then:
   git remote add origin https://github.com/YOUR_USERNAME/alzheimers-backend.git
   git push -u origin main
   ```

2. **Deploy on Render:**
   - Go to [render.com](https://render.com)
   - New → Web Service
   - Connect GitHub repo
   - Settings:
     - **Build**: `pip install -r requirements.txt`
     - **Start**: `uvicorn src.api:app --host 0.0.0.0 --port $PORT`
     - **Plan**: Free
   - Add env var: `ENVIRONMENT=production`
   - Deploy and **copy your URL**: `https://YOUR-APP.onrender.com`

---

## Step 2: Deploy Frontend (5 minutes)

1. **Push frontend to GitHub:**
   ```bash
   cd frontend
   git init
   git add .
   git commit -m "Ready for deployment"
   # Create repo on GitHub, then:
   git remote add origin https://github.com/YOUR_USERNAME/alzheimers-frontend.git
   git push -u origin main
   ```

2. **Deploy on Netlify:**
   - Go to [netlify.com](https://netlify.com)
   - Add new site → Import from Git
   - Connect GitHub → Select repo
   - Build settings:
     - **Build command**: `npm run build`
     - **Publish directory**: `dist`
   - **Add environment variable:**
     - Key: `VITE_API_URL`
     - Value: `https://YOUR-APP.onrender.com` (from Step 1)
   - Deploy and **copy your URL**: `https://YOUR-APP.netlify.app`

---

## Step 3: Connect Them (1 minute)

1. **Go back to Render:**
   - Your service → Environment
   - Add variable: `FRONTEND_URL`
   - Value: `https://YOUR-APP.netlify.app` (from Step 2)
   - Save (auto-redeploys)

2. **Test:**
   - Visit your Netlify URL
   - Fill form → Click Analyze
   - Should work! 🎉

---

## ✅ Done!

Your app is live! Total time: ~10 minutes.

**Full guide:** See `DEPLOYMENT_STEPS.md` for detailed instructions.

