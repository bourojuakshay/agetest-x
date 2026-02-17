# Railway Build Troubleshooting Guide

## Common Build Failures & Solutions

### Issue 1: Requirements.txt Format Error

**Symptom**: Build fails during pip install
**Cause**: `--index-url` on same line as package
**Solution**: ✅ Fixed in latest commit

Changed from:
```
torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
```

To:
```
--extra-index-url https://download.pytorch.org/whl/cpu
torch>=2.0.0
torchvision>=0.15.0
```

Also changed `opencv-python` to `opencv-python-headless` (better for server environments).

---

### Issue 2: Large Model File

**Symptom**: Build timeout or memory issues
**Cause**: `age_model.pth` is 43MB
**Status**: ✅ File is in git, should work

**If build still fails**:
- Railway free tier has memory limits
- Consider upgrading to Railway Pro for larger builds

---

### Issue 3: Python Version Mismatch

**Current**: `nixpacks.toml` specifies Python 3.10
**Check**: Ensure your local Python version is compatible

---

### Issue 4: Missing System Dependencies

**For OpenCV**: Requires system libraries

**Solution**: Update `nixpacks.toml` to include system packages:

```toml
[phases.setup]
nixPkgs = ["python310", "libGL", "glib"]
```

---

## What to Check in Railway Logs

Look for these specific errors:

### 1. Pip Installation Errors
```
ERROR: Invalid requirement
ERROR: Could not find a version that satisfies
```
**Fix**: Check requirements.txt format

### 2. Memory Errors
```
Killed
Out of memory
```
**Fix**: Upgrade Railway plan or reduce model size

### 3. Missing Dependencies
```
ImportError: libGL.so.1
```
**Fix**: Add system packages to nixpacks.toml

### 4. Port Binding Errors
```
Address already in use
```
**Fix**: Ensure using `$PORT` environment variable

---

## Current Configuration Status

✅ Root directory set to: `backend`
✅ Model file in git: `age_model.pth` (43MB)
✅ Requirements.txt: **FIXED** (latest commit)
✅ nixpacks.toml: Configured
✅ railway.toml: Configured

---

## Next Steps

1. **Pull latest changes** (requirements.txt was just fixed):
   ```bash
   git pull origin main
   ```

2. **Retry deployment** in Railway:
   - Railway should auto-deploy after git push
   - Or manually trigger redeploy in dashboard

3. **Monitor build logs** for specific errors

4. **If still failing**, share the exact error message from Railway logs

---

## Alternative: Simplified Requirements

If build still fails, try this minimal `requirements.txt`:

```
fastapi==0.100.0
uvicorn==0.23.0
numpy==1.24.3
pillow==10.0.0
python-multipart==0.0.6
opencv-python-headless==4.8.1.78
cryptography==41.0.3
torch==2.0.1
torchvision==0.15.2
```

(Pinned versions for reproducibility)

---

## Railway-Specific Tips

1. **Build Time**: First build may take 5-10 minutes
2. **Caching**: Subsequent builds are faster
3. **Logs**: Always check full build logs in Railway dashboard
4. **Memory**: Free tier has 512MB RAM limit
5. **Timeout**: Build timeout is usually 10 minutes

---

## If All Else Fails

**Option 1**: Deploy without the model file initially
- Remove `age_model.pth` temporarily
- Deploy to verify everything else works
- Add model file later

**Option 2**: Use Railway Pro
- More memory and build time
- Better for ML applications

**Option 3**: Alternative deployment
- Use Render.com or Fly.io
- Similar process, different platform
