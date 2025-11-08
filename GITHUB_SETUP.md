# GitHub Repository Setup Guide

Follow these steps to push your Terrain Recognition System to GitHub.

## Prerequisites

1. Git installed on your system
2. GitHub account
3. Project files ready

## Step 1: Initialize Git Repository (if not already done)

```bash
cd terrain-recognisation
git init
```

## Step 2: Add All Files

```bash
git add .
```

## Step 3: Create Initial Commit

```bash
git commit -m "Initial commit: Deep Learning Terrain Recognition System with Dashboard"
```

## Step 4: Create GitHub Repository

1. Go to [GitHub.com](https://github.com)
2. Click the "+" icon in the top right
3. Select "New repository"
4. Name it: `terrain-recognition-system` (or your preferred name)
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

## Step 5: Add Remote and Push

```bash
# Add your GitHub repository as remote (replace USERNAME with your GitHub username)
git remote add origin https://github.com/USERNAME/terrain-recognition-system.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

## Step 6: Verify Upload

1. Visit your repository on GitHub
2. Check that all files are present
3. Verify README.md displays correctly

## Important Notes

### Files Not Tracked (by design)

The following files are in `.gitignore` and won't be uploaded:
- `terrain_model.h5` (large model file)
- `prediction_history.json` (user data)
- `class_indices.json` (generated file)
- `training_history.png` (generated file)
- `static/uploads/*` (uploaded images)
- Virtual environment folders

### If You Want to Include the Model

If you want to share your trained model on GitHub:

1. Remove or comment out this line in `.gitignore`:
   ```
   # terrain_model.h5
   ```

2. Note: GitHub has a 100MB file size limit. If your model is larger, consider:
   - Using Git LFS (Large File Storage)
   - Hosting the model on a cloud storage service
   - Using a model hosting platform

### Using Git LFS for Large Files

If your model file is large:

```bash
# Install Git LFS
git lfs install

# Track .h5 files
git lfs track "*.h5"

# Add .gitattributes
git add .gitattributes

# Commit and push
git add terrain_model.h5
git commit -m "Add trained model with Git LFS"
git push
```

## Updating the Repository

After making changes:

```bash
# Add changes
git add .

# Commit with a message
git commit -m "Description of changes"

# Push to GitHub
git push
```

## Repository Description

Add this description to your GitHub repository:

```
🌍 Deep Learning Terrain Recognition System - Classify terrain types from images using CNN. Features interactive dashboard, real-time predictions, and data visualizations.
```

## Topics/Tags

Add these topics to your repository:
- `deep-learning`
- `tensorflow`
- `keras`
- `flask`
- `cnn`
- `computer-vision`
- `image-classification`
- `terrain-recognition`
- `machine-learning`
- `python`

## License

Consider adding a LICENSE file. Common options:
- MIT License (permissive)
- Apache 2.0 License
- GNU GPL v3

## Badges

Add these badges to your README.md (optional):

```markdown
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
```

## Deployment

Consider deploying your application:

1. **Heroku**: Easy deployment for Flask apps
2. **Railway**: Simple deployment platform
3. **Render**: Free tier available
4. **AWS/GCP/Azure**: For production deployment

## Support

If you encounter issues:
1. Check the README.md for setup instructions
2. Review the error messages
3. Check GitHub Issues (if any)
4. Create a new issue on GitHub

---

Happy Coding! 🚀

