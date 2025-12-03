# Backend Deployment Quick Reference

## Environment Variables (Set in Render Dashboard)

| Variable | Value | Description |
|----------|-------|-------------|
| `ENVIRONMENT` | `production` | Set environment to production |
| `FRONTEND_URL` | `https://your-frontend.netlify.app` | Your frontend URL (set after frontend deploys) |
| `MODEL_PATH` | (optional) | Custom path to model file |

## Important Files

- `Procfile` - Tells Render how to run the app
- `runtime.txt` - Python version
- `requirements.txt` - Python dependencies
- `models/best_model.pkl` - Trained model (must be in Git repository)

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn src.api:app --reload

# API will be available at http://127.0.0.1:8000
```

## Render Deployment Checklist

- [ ] Push code to GitHub
- [ ] Connect GitHub repo to Render
- [ ] Set `ENVIRONMENT=production` in Render
- [ ] Ensure `models/best_model.pkl` is in repository
- [ ] Deploy and note backend URL
- [ ] Set `FRONTEND_URL` after frontend deploys
- [ ] Redeploy backend to apply CORS changes

