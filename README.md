# Website

This website is built using [Docusaurus](https://docusaurus.io/), a modern static website generator.

## Installation

Use **npm** (same as CI). From the repo root:

```bash
npm ci
```

For day-to-day work, a normal install is fine:

```bash
npm install
```

After you change dependencies in `package.json`, run `npm install` and commit the updated **`package-lock.json`**.

## Local Development

```bash
npm start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

## Build

```bash
npm run build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

## Deployment

### Automatic Deployment (Recommended)

The site is automatically deployed to GitHub Pages using GitHub Actions when you push to the `main` or `master` branch. The workflow will:
1. Build the Docusaurus site
2. Deploy to the `gh-pages` branch
3. GitHub Pages will serve the site from the `gh-pages` branch

**Important**: Make sure GitHub Pages is configured in your repository settings:
- Go to Settings → Pages
- Source: Deploy from a branch
- Branch: `gh-pages` / `root`

### Manual Deployment

Using SSH:

```bash
USE_SSH=true npm run deploy
```

Not using SSH:

```bash
GIT_USER=type1compute npm run deploy
```

### Configuration Notes

- **Repository**: `type1compute/docs`
- **Base URL**: `/docs/` (configured in `docusaurus.config.js`)
- **GitHub Pages URL**: `https://type1compute.github.io/docs/`

If your repository name is different, update the `baseUrl` in `docusaurus.config.js`:
- Repository `type1compute/docs` → `baseUrl: '/docs/'`
- Repository `type1compute/pages` → `baseUrl: '/pages/'`
- User/organization site → `baseUrl: '/'`
