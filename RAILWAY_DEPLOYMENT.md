# Railway Deployment Guide

## Prerequisites
- Railway account (sign up at https://railway.app)
- Git repository with all changes committed
- Railway CLI installed (optional, can use web dashboard)

## Step-by-Step Deployment

### Option 1: Deploy via Railway Dashboard (Recommended)

#### 1. Deploy Backend Service

1. **Login to Railway**: Go to https://railway.app and sign in
2. **Create New Project**: Click "New Project"
3. **Deploy from GitHub**: 
   - Select "Deploy from GitHub repo"
   - Choose your repository: `bourojuakshay/agetest-x`
   
4. **IMPORTANT - Configure Root Directory**:
   - After selecting the repo, Railway will try to deploy
   - Click on the service settings (gear icon)
   - Go to "Settings" tab
   - Scroll to "Root Directory"
   - **Set Root Directory to: `backend`** ⚠️ This is critical!
   - Click "Save"
   - The service will redeploy automatically

5. **Alternative Method - Set Before Deploy**:
   - When creating the service, look for "Configure" or "Settings"
   - Set the root directory to `backend` before first deployment
   
6. **Set Environment Variables** (optional):
   - Go to service "Variables" tab
   - Add `ENVIRONMENT=production`
   - Add `ALLOWED_ORIGINS` (will set after frontend deploys)

7. **Get Backend URL**:
   - After deployment succeeds, go to "Settings" → "Domains"
   - Copy the public URL (e.g., `https://your-backend.up.railway.app`)
   - Save this URL for frontend configuration

#### 2. Deploy Frontend Service

1. **Add Another Service**:
   - In the same Railway project, click "New Service"
   - Select "GitHub Repo" again
   - Choose the same repository

2. **Configure Frontend Service**:
   - Set **Root Directory**: `frontend`
   - Railway will detect the Dockerfile

3. **Update Frontend Configuration**:
   - Before deploying, update `frontend/config.js`:
   - Replace `REPLACE_WITH_RAILWAY_BACKEND_URL` with your backend URL from step 1
   - Commit and push this change:
   ```bash
   git add frontend/config.js
   git commit -m "Update backend URL for Railway deployment"
   git push origin main
   ```

4. **Deploy Frontend**:
   - Railway will automatically deploy after the push
   - Get the frontend URL (e.g., `https://your-frontend.railway.app`)

#### 3. Update Backend CORS

1. **Set ALLOWED_ORIGINS**:
   - Go to backend service → Variables
   - Add: `ALLOWED_ORIGINS=https://your-frontend.railway.app`
   - Backend will restart automatically

2. **Verify CORS**:
   - Visit your frontend URL
   - Check browser console for CORS errors
   - Should be able to communicate with backend

---

### Option 2: Deploy via Railway CLI

#### 1. Install Railway CLI
```bash
npm install -g @railway/cli
```

#### 2. Login
```bash
railway login
```

#### 3. Initialize Project
```bash
cd /Users/bouroju/Downloads/agetest-main
railway init
```

#### 4. Deploy Backend
```bash
cd backend
railway up
# Note the deployment URL
```

#### 5. Update Frontend Config
Edit `frontend/config.js` with the backend URL, then:
```bash
git add frontend/config.js
git commit -m "Update backend URL"
git push origin main
```

#### 6. Deploy Frontend
```bash
cd ../frontend
railway up
# Note the deployment URL
```

#### 7. Set Environment Variables
```bash
# For backend service
railway variables set ENVIRONMENT=production
railway variables set ALLOWED_ORIGINS=https://your-frontend.railway.app
```

---

## Post-Deployment Checklist

### 1. Verify Backend
- [ ] Visit `https://your-backend.railway.app/docs`
- [ ] Swagger UI should load
- [ ] Test the `/api/age-check` endpoint

### 2. Verify Frontend
- [ ] Visit `https://your-frontend.railway.app`
- [ ] Page loads without errors
- [ ] Check browser console (F12) for errors

### 3. Test End-to-End
- [ ] Allow camera access
- [ ] Age detection should work
- [ ] Content should filter based on detected age
- [ ] No CORS errors in console

### 4. Monitor Logs
- [ ] Check Railway dashboard for both services
- [ ] Monitor for errors or crashes
- [ ] Check resource usage

---

## Important Configuration Files Created

1. **backend/railway.toml** - Backend deployment config
2. **frontend/Dockerfile** - Frontend container config
3. **frontend/nginx.conf** - Nginx web server config
4. **frontend/config.js** - Environment-based API URL
5. **railway.json** - Root project config

---

## Troubleshooting

### CORS Errors
- Ensure `ALLOWED_ORIGINS` includes your frontend URL
- Check backend logs for CORS-related errors
- Verify frontend is using HTTPS (Railway provides this automatically)

### Backend Not Starting
- Check Railway logs for Python errors
- Verify `age_model.pth` is in the repository (44MB)
- Ensure all dependencies in `requirements.txt` are correct

### Frontend Not Loading
- Check nginx logs in Railway dashboard
- Verify Dockerfile builds successfully
- Check that all static files are present

### Camera Not Working
- HTTPS is required for camera access (Railway provides this)
- User must grant camera permissions
- Check browser compatibility

---

## Cost Optimization

Railway free tier includes:
- $5 credit per month
- Enough for small projects

To optimize costs:
1. Use Railway's sleep feature for inactive services
2. Monitor resource usage in dashboard
3. Consider upgrading if you exceed free tier

---

## Next Steps After Deployment

1. **Custom Domain** (Optional):
   - Add your own domain in Railway settings
   - Update CORS and config accordingly

2. **Monitoring**:
   - Set up Railway alerts
   - Monitor error rates and performance

3. **Scaling**:
   - Adjust replicas if needed
   - Monitor response times

4. **Security**:
   - Rotate encryption keys regularly
   - Review Firebase security rules
   - Monitor for suspicious activity
