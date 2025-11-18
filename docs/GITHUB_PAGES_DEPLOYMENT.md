# Deploying Ternary Computing Sandbox to GitHub Pages

This guide explains how to deploy the interactive ternary computing sandbox to GitHub Pages so anyone can use it via `https://yourusername.github.io/trit-computing-machine/`.

## Prerequisites

- Git installed
- GitHub account
- Repository cloned locally

## Deployment Steps

### Option 1: Deploy from `docs/web-sandbox` directory (Recommended)

This is the simplest method as the files are already in the `docs/web-sandbox` directory.

1. **Push your code to GitHub:**

```bash
# Make sure all files are committed
git add docs/web-sandbox/
git commit -m "Add interactive web sandbox"
git push origin main
```

2. **Enable GitHub Pages:**

   - Go to your repository on GitHub: `https://github.com/v-like-engine/trit-computing-machine`
   - Click on **Settings** (gear icon)
   - Scroll down to **Pages** section (left sidebar)
   - Under **Source**, select **Branch: main**
   - Under **Folder**, select **/docs**
   - Click **Save**

3. **Wait for deployment:**

   - GitHub will build and deploy your site (takes 1-2 minutes)
   - A green checkmark will appear when ready
   - Your site will be available at: `https://v-like-engine.github.io/trit-computing-machine/web-sandbox/`

4. **Optional: Set custom domain or redirect:**

   To make the sandbox the main page, you can:
   - Rename `docs/web-sandbox/index.html` to `docs/index.html`
   - Or create a redirect in `docs/index.html`:

```html
<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="refresh" content="0; url=web-sandbox/index.html">
</head>
<body>
    <p>Redirecting to sandbox...</p>
</body>
</html>
```

### Option 2: Deploy using GitHub Actions (Advanced)

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs/web-sandbox
          publish_branch: gh-pages
```

Then enable GitHub Pages from the `gh-pages` branch.

### Option 3: Deploy entire `docs` folder

If you want to include documentation alongside the sandbox:

1. **Configure GitHub Pages:**
   - Settings → Pages → Source: **main** → Folder: **/docs**

2. **Create `docs/index.html`:**

```html
<!DOCTYPE html>
<html>
<head>
    <title>Ternary Computing Machine</title>
</head>
<body>
    <h1>Ternary Computing Machine</h1>
    <ul>
        <li><a href="web-sandbox/index.html">Interactive Sandbox</a></li>
        <li><a href="NEURAL_NETWORKS.md">Neural Networks Documentation</a></li>
        <li><a href="IDEAS.md">Ideas and Applications</a></li>
    </ul>
</body>
</html>
```

3. **Your site structure will be:**
   - `https://yourusername.github.io/trit-computing-machine/` - Index
   - `https://yourusername.github.io/trit-computing-machine/web-sandbox/` - Sandbox

## Accessing the Deployed Sandbox

Once deployed, users can:

1. **Visit the URL:**
   ```
   https://v-like-engine.github.io/trit-computing-machine/web-sandbox/
   ```

2. **Use the sandbox:**
   - Evaluate ternary logic expressions
   - Build and train neural networks
   - Compare encoding methods
   - Load datasets (CSV or synthetic MNIST)

3. **No installation required:**
   - Runs entirely in the browser
   - No Python or backend needed
   - Works on any device with a modern browser

## Customization

### Update Site Title and Meta Tags

Edit `docs/web-sandbox/index.html`:

```html
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Interactive ternary computing sandbox with neural networks">
    <meta name="keywords" content="ternary computing, balanced ternary, neural networks, machine learning">
    <meta name="author" content="Your Name">
    <title>Ternary Computing Sandbox</title>
    <link rel="stylesheet" href="styles.css">
</head>
```

### Add Google Analytics (Optional)

Before `</head>`:

```html
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=YOUR-GA-ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'YOUR-GA-ID');
</script>
```

### Add Custom Domain

1. Add `CNAME` file in `docs/web-sandbox/` (or `docs/`):
   ```
   ternary.yourdomain.com
   ```

2. Configure DNS:
   - Add CNAME record pointing to `yourusername.github.io`

3. Enable HTTPS in GitHub Pages settings

## Troubleshooting

### Page not loading

- **Check GitHub Pages settings**: Make sure Pages is enabled
- **Wait a few minutes**: Initial deployment takes time
- **Clear browser cache**: Force refresh with Ctrl+Shift+R (or Cmd+Shift+R)

### 404 Error

- **Verify branch**: Make sure you selected the correct branch
- **Check folder structure**: Files should be in `docs/web-sandbox/`
- **File names**: Must be `index.html`, not `Index.html` (case-sensitive on some systems)

### JavaScript not working

- **Check console**: Open browser DevTools (F12) and look for errors
- **File paths**: Make sure all `<script src="...">` paths are correct
- **CORS errors**: If loading external resources, check CORS policies

### Styling issues

- **Check CSS path**: Verify `<link rel="stylesheet" href="styles.css">`
- **Relative paths**: Use relative paths, not absolute
- **Cache**: Clear browser cache

## Local Testing Before Deployment

Test locally before deploying:

### Method 1: Python HTTP Server

```bash
cd docs/web-sandbox
python3 -m http.server 8000
```

Visit: `http://localhost:8000`

### Method 2: VS Code Live Server

1. Install "Live Server" extension
2. Right-click `index.html`
3. Select "Open with Live Server"

### Method 3: Node.js HTTP Server

```bash
npx http-server docs/web-sandbox -p 8000
```

## Updating the Sandbox

To update the live site:

```bash
# Make changes to files in docs/web-sandbox/
# Test locally first!

git add docs/web-sandbox/
git commit -m "Update sandbox: [describe changes]"
git push origin main

# GitHub Pages will auto-deploy in 1-2 minutes
```

## Security Considerations

The sandbox runs entirely client-side, so:

✅ **Safe:**
- All computations happen in browser
- No server-side code execution
- No data sent to servers
- User data stays local

⚠️ **Be aware:**
- Large datasets may slow down browser
- Limited by browser memory (~2GB typical)
- No authentication/user accounts

## Performance Tips

1. **Optimize for mobile:**
   - Test on various screen sizes
   - Use responsive design (already implemented)
   - Limit animation/heavy computations

2. **Reduce file sizes:**
   - Minify JavaScript (optional)
   - Compress images (if added)
   - Use CDN for large libraries

3. **Cache static assets:**
   - Add service worker (advanced)
   - Use browser caching headers

## Sharing the Sandbox

Share your sandbox:

- **Direct link:** `https://v-like-engine.github.io/trit-computing-machine/web-sandbox/`
- **QR Code:** Generate a QR code pointing to the URL
- **Social media:** Share with #TernaryComputing #BalancedTernary
- **Embed:** Use `<iframe>` to embed in other sites

Example embed:

```html
<iframe src="https://v-like-engine.github.io/trit-computing-machine/web-sandbox/"
        width="100%"
        height="800px"
        frameborder="0">
</iframe>
```

## Next Steps

After deployment:

1. ✅ Test all features on the live site
2. ✅ Share with the community
3. ✅ Collect feedback
4. ✅ Add new features based on usage
5. ✅ Monitor performance and errors

## Support

If you encounter issues:

1. Check [GitHub Issues](https://github.com/v-like-engine/trit-computing-machine/issues)
2. Review [GitHub Pages docs](https://docs.github.com/en/pages)
3. Test locally first to isolate deployment vs code issues

---

**Congratulations!** Your ternary computing sandbox is now live and accessible to the world! 🎉
