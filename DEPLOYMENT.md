# Deployment Guide

## Fixing README.md Showing Instead of Site

If you're seeing the README.md file instead of your Docusaurus site on GitHub Pages, follow these steps:

### 1. Verify Repository Name

Check your repository name. The `baseUrl` in `docusaurus.config.js` must match:

- Repository: `type1compute/docs` → `baseUrl: '/docs/'`
- Repository: `type1compute/pages` → `baseUrl: '/pages/'`
- Repository: `type1compute/type1compute.github.io` → `baseUrl: '/'` (user/organization site)

### 2. Configure GitHub Pages Settings

1. Go to your repository on GitHub
2. Navigate to **Settings** → **Pages**
3. Under **Source**, select:
   - **Branch**: `gh-pages`
   - **Folder**: `/ (root)`
4. Click **Save**

### 3. Set Up Personal Access Token (Required if Read/Write Permissions Disabled)

If your repository has "Read and write permissions" disabled, you need to create a Personal Access Token:

1. **Create a Personal Access Token (PAT)**:
   - Go to GitHub → **Settings** → **Developer settings** → **Personal access tokens** → **Tokens (classic)**
   - Click **Generate new token (classic)**
   - Give it a name like "GitHub Pages Deploy"
   - Select scopes:
     - ✅ `repo` (Full control of private repositories)
     - ✅ `workflow` (Update GitHub Action workflows)
   - Click **Generate token**
   - **Copy the token immediately** (you won't see it again!)

2. **Add Token as Secret**:
   - Go to your repository → **Settings** → **Secrets and variables** → **Actions**
   - Click **New repository secret**
   - Name: `PERSONAL_ACCESS_TOKEN`
   - Value: Paste your token
   - Click **Add secret**

3. **The workflow will automatically use the PAT** if `GITHUB_TOKEN` doesn't have write permissions.

### 4. Deploy Using GitHub Actions (Recommended)

The repository includes a GitHub Actions workflow (`.github/workflows/deploy.yml`) that automatically:
- Builds the site when you push to `main` or `master`
- Deploys to the `gh-pages` branch
- Handles the `.nojekyll` file automatically

**To use this:**
1. Push your code to the `main` or `master` branch
2. The workflow will run automatically
3. Check the **Actions** tab to see the deployment status
4. Wait a few minutes for GitHub Pages to update

### 5. Manual Deployment

If you prefer manual deployment:

```bash
# Build the site
npm run build

# Deploy to gh-pages branch
GIT_USER=type1compute npm run deploy
```

Or with SSH:

```bash
USE_SSH=true npm run deploy
```

### 6. Verify Deployment

After deployment:

1. Check the `gh-pages` branch exists and contains the `build` folder contents
2. Verify `.nojekyll` file is in the root of `gh-pages` branch
3. Wait 1-2 minutes for GitHub Pages to rebuild
4. Visit: `https://type1compute.github.io/docs/` (adjust URL based on your repo name)

### 7. Troubleshooting

**Issue: Still seeing README.md**
- ✅ Check `baseUrl` matches repository name
- ✅ Verify GitHub Pages is set to serve from `gh-pages` branch
- ✅ Ensure `.nojekyll` file exists in `gh-pages` branch root
- ✅ Clear browser cache and try incognito mode
- ✅ Wait a few minutes for GitHub Pages to update

**Issue: Permission Denied (403 Error)**
- ✅ Create a Personal Access Token (PAT) with `repo` scope
- ✅ Add PAT as `PERSONAL_ACCESS_TOKEN` secret in repository settings
- ✅ The workflow will automatically use PAT if GITHUB_TOKEN lacks permissions
- ✅ Verify the token has not expired

**Issue: 404 errors**
- ✅ Check `baseUrl` is correct (should end with `/`)
- ✅ Verify all assets are in the `build` folder
- ✅ Check browser console for 404 errors on specific files

**Issue: Styles not loading**
- ✅ Check `baseUrl` is correct
- ✅ Verify all static assets are deployed
- ✅ Check network tab for failed resource loads

### 8. Current Configuration

```javascript
// docusaurus.config.js
url: 'https://type1compute.github.io',
baseUrl: '/docs/',
organizationName: 'type1compute',
projectName: 'docs',
```

If your repository name is different, update these values accordingly.
