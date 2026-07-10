# GitHub Repository Setup Guide

## Quick Setup (3 Steps)

### Step 1: Create Repository on GitHub

1. Go to: https://github.com/new
2. Fill in the details:
   - **Repository name:** `customer-transaction-prediction`
   - **Description:** `PRCP-1003: Complete ML project for customer transaction prediction with Streamlit app`
   - **Visibility:** Public (recommended) or Private
   - **DO NOT** check any of these:
     - ❌ Add a README file
     - ❌ Add .gitignore
     - ❌ Choose a license
   - Click **Create repository**

### Step 2: Push Code to GitHub

Open your terminal/command prompt and run:

```bash
cd "c:\Users\arock\Downloads\Datamites Project\PRCP-1003 Customer Transaction Prediction"
git push -u origin main
```

**If prompted for credentials:**
- **Username:** arockiadassdev
- **Password:** Use your GitHub Personal Access Token (not your password)
  - If you don't have a token, create one at: https://github.com/settings/tokens
  - Select scopes: `repo` (full control of private repositories)

### Step 3: Verify

Visit your repository: https://github.com/arockiadassdev/customer-transaction-prediction

You should see all your project files!

---

## Alternative: Using GitHub Desktop

If you prefer a GUI:

1. Download GitHub Desktop: https://desktop.github.com/
2. Open your project folder
3. Click "Publish repository"
4. Follow the prompts

---

## Troubleshooting

### Error: "Repository not found"
- Make sure you created the repository on GitHub first
- Check the repository name matches exactly

### Error: "Invalid username or password"
- Use a Personal Access Token instead of password
- Enable 2FA on your GitHub account for better security

### Large files not uploading
- The model files (.pkl) are already tracked
- If needed, use Git LFS: `git lfs install`

---

## Your Repository Details

- **Username:** arockiadassdev
- **Email:** arockiadassdev@gmail.com
- **Repository URL:** https://github.com/arockiadassdev/customer-transaction-prediction.git
- **Branch:** main

---

## What Gets Pushed

✅ All source code (src/)
✅ Streamlit app (app/)
✅ Jupyter notebook (notebooks/)
✅ Model artifacts (models/)
✅ Documentation (README.md, LICENSE, .gitignore)
✅ Requirements (requirements.txt)
❌ Data files (ignored by .gitignore)
❌ IDE files (ignored by .gitignore)

---

## Next Steps After Push

1. **Enable GitHub Pages** (optional): Settings → Pages → Source: main branch
2. **Add topics:** machine-learning, streamlit, classification, data-science
3. **Pin the repository** on your GitHub profile
4. **Share on LinkedIn** with a post about your project!

---

Need help? Contact: arockiadassdev@gmail.com