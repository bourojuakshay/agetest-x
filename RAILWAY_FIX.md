# Railway Deployment - Quick Fix Guide

## ⚠️ Issue: Nixpacks Build Failed

**Error**: "Nixpacks was unable to generate a build plan for this app"

**Cause**: Railway is trying to build from the root directory, but your code is in `backend/` and `frontend/` subdirectories.

---

## ✅ Solution: Set Root Directory

### For Backend Service

**Method 1: Via Service Settings (After Creation)**

1. In Railway dashboard, click on your service
2. Click the **Settings** tab (gear icon)
3. Scroll down to **"Root Directory"** or **"Source"** section
4. Enter: `backend`
5. Click **"Save"** or **"Update"**
6. Service will automatically redeploy

**Method 2: During Service Creation**

1. When adding a new service from GitHub
2. Look for **"Configure"** or **"Advanced Settings"**
3. Set **Root Directory** to `backend` before deploying

---

### For Frontend Service

Same process, but set **Root Directory** to: `frontend`

---

## Step-by-Step Visual Guide

### Backend Deployment

```
Railway Dashboard
  └─ Your Project
      └─ New Service
          └─ GitHub Repo (bourojuakshay/agetest-x)
              └─ Settings ⚙️
                  └─ Root Directory: backend  ← SET THIS!
                      └─ Save
```

### What Should Happen

After setting the root directory:
- ✅ Railway will detect `backend/nixpacks.toml`
- ✅ Railway will find `backend/requirements.txt`
- ✅ Railway will detect Python app
- ✅ Build will succeed

---

## Alternative: Using Railway CLI

If you prefer using the CLI:

```bash
# Navigate to your project
cd /Users/bouroju/Downloads/agetest-main

# Login to Railway
railway login

# Link to your project (if not already linked)
railway link

# Deploy backend with root directory
cd backend
railway up

# Deploy frontend with root directory
cd ../frontend
railway up
```

The CLI automatically uses the current directory as the root.

---

## Verification

### Backend Deployment Success Indicators:
- ✅ Build logs show: "Installing Python dependencies"
- ✅ Build logs show: "Installing packages from requirements.txt"
- ✅ Deployment shows: "uvicorn app:app --host 0.0.0.0 --port $PORT"
- ✅ Service status: "Active" or "Running"
- ✅ Can access: `https://your-backend.up.railway.app/docs`

### Frontend Deployment Success Indicators:
- ✅ Build logs show: "Building Dockerfile"
- ✅ Build logs show: "nginx:alpine"
- ✅ Service status: "Active" or "Running"
- ✅ Can access: `https://your-frontend.up.railway.app`

---

## Common Issues

### Issue: "No such file or directory: requirements.txt"
**Fix**: Root directory not set correctly. Ensure it's set to `backend`

### Issue: "Dockerfile not found"
**Fix**: For frontend, ensure root directory is set to `frontend`

### Issue: Service keeps failing
**Fix**: Check the deployment logs in Railway dashboard for specific errors

---

## Next Steps After Successful Deployment

1. **Get Backend URL** from Railway dashboard
2. **Update** `frontend/config.js`:
   ```javascript
   production: {
       API_URL: "https://your-backend.up.railway.app/api/age-check"
   }
   ```
3. **Commit and push** the change
4. **Deploy frontend** (will auto-deploy after push)
5. **Set CORS** in backend environment variables:
   - `ALLOWED_ORIGINS=https://your-frontend.up.railway.app`

---

## Need Help?

If you're still having issues:
1. Check Railway deployment logs
2. Verify root directory is set correctly
3. Ensure all files are committed to git
4. Check that `backend/age_model.pth` exists (43MB)

---

**TL;DR**: In Railway service settings, set **Root Directory** to `backend` for backend service and `frontend` for frontend service. That's it! 🚀
