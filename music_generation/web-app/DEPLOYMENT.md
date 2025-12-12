# 🚀 Deployment Guide for AI Music Generator

## Quick Deploy to Vercel (Recommended)

### Prerequisites
- GitHub account
- Vercel account (free tier works great!)
- Your code pushed to a GitHub repository

### Method 1: Deploy via Vercel Dashboard (Easiest)

1. **Push to GitHub**
   ```bash
   cd /Users/martinbattu/Documents/CST435_FinalProj
   git add music_generation/web-app
   git commit -m "Add AI Music Generator web app"
   git push origin main
   ```

2. **Connect to Vercel**
   - Go to [vercel.com](https://vercel.com)
   - Click "Add New Project"
   - Import your GitHub repository
   - Select the `music_generation/web-app` directory as the root directory

3. **Configure Build Settings** (Auto-detected)
   - Framework Preset: **Next.js**
   - Build Command: `npm run build`
   - Output Directory: `.next`
   - Install Command: `npm install`

4. **Deploy**
   - Click "Deploy"
   - Wait 2-3 minutes
   - Your app is live! 🎉

### Method 2: Deploy via Vercel CLI

```bash
# Install Vercel CLI globally
npm install -g vercel

# Navigate to your web app
cd /Users/martinbattu/Documents/CST435_FinalProj/music_generation/web-app

# Login to Vercel
vercel login

# Deploy (production)
vercel --prod
```

### Method 3: One-Click Deploy

Add this button to your README:

```markdown
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=YOUR_REPO_URL&project-name=ai-music-generator)
```

## 🎯 What Happens During Deployment

1. **Build Phase**
   - Installs dependencies (`npm install`)
   - Compiles TypeScript
   - Builds Next.js app
   - Optimizes assets

2. **Deploy Phase**
   - Uploads to Vercel CDN
   - Configures serverless functions
   - Enables HTTPS automatically
   - Sets up global edge network

## ⚙️ Configuration

### Environment Variables (Optional)

If you need environment variables, add them in Vercel Dashboard:
- Go to Project Settings → Environment Variables
- Currently, this app doesn't need any!

### Custom Domain (Optional)

1. Go to Project Settings → Domains
2. Add your custom domain
3. Update DNS records as instructed
4. SSL certificate is automatic

## 🧪 Testing Before Deploy

### Local Production Build

```bash
cd music_generation/web-app

# Build
npm run build

# Test production build locally
npm start

# Open http://localhost:3000
```

### Check Build Size

```bash
npm run build

# Look for the output showing bundle sizes
# Should be around:
# - Page Size: ~500-800 KB
# - First Load JS: ~5-6 MB (TensorFlow.js is large)
```

## 📊 Performance Optimization

### Current Setup
- ✅ Automatic code splitting
- ✅ Image optimization
- ✅ Static generation
- ✅ Edge caching
- ✅ Gzip compression

### Optional Improvements

1. **Reduce TensorFlow.js Bundle**
   - Use `@tensorflow/tfjs-core` instead of full package
   - Load model weights separately

2. **Enable Incremental Static Regeneration**
   ```typescript
   export const revalidate = 3600; // Revalidate every hour
   ```

3. **Add Analytics**
   ```bash
   npm install @vercel/analytics
   ```

## 🐛 Troubleshooting

### Build Fails

**Error: TypeScript errors**
```bash
# Run type check locally
npm run build

# Fix any TypeScript errors shown
```

**Error: Out of memory**
```json
// In package.json, update build script:
"build": "NODE_OPTIONS='--max_old_space_size=4096' next build"
```

### Runtime Issues

**Audio doesn't play**
- Check browser console for errors
- Ensure Tone.js is loaded
- Click page first (browsers require user interaction for audio)

**Model loads slowly**
- Normal! TensorFlow.js is ~4MB
- Consider adding a loading screen
- Model initializes once per session

**MIDI download fails**
- Check browser console
- Verify Blob creation
- Ensure MIDI encoder works in dev

## 📱 Browser Compatibility

✅ Supported:
- Chrome/Edge (95+)
- Firefox (90+)
- Safari (15+)
- Mobile browsers (iOS 15+, Android Chrome)

⚠️ Limited Support:
- Safari < 15 (WebAudio issues)
- IE 11 (not supported)

## 🔒 Security

### Current Setup
- ✅ HTTPS enforced
- ✅ CSP headers (default Next.js)
- ✅ XSS protection
- ✅ No sensitive data stored

### Recommendations
- Enable CORS if needed
- Add rate limiting for API routes (if added)
- Monitor Vercel analytics for abuse

## 💰 Costs

### Vercel Free Tier Includes:
- ✅ 100 GB bandwidth/month
- ✅ Unlimited deployments
- ✅ HTTPS/SSL certificates
- ✅ Preview deployments for PRs
- ✅ Serverless functions (up to 100 GB-hrs)

### This App Usage:
- ~5-6 MB per user visit (first load)
- ~500 KB for returning users (cached)
- No serverless functions used
- Estimate: **~1,600 users/month on free tier**

## 📈 Monitoring

### Vercel Analytics (Built-in)
- Real-time visitor count
- Performance metrics
- Error tracking
- Geographic distribution

### Add Advanced Analytics
```bash
npm install @vercel/analytics

# In app/layout.tsx
import { Analytics } from '@vercel/analytics/react';

export default function RootLayout({ children }) {
  return (
    <html>
      <body>
        {children}
        <Analytics />
      </body>
    </html>
  );
}
```

## 🚀 Post-Deployment

### Share Your App
- Copy the Vercel URL (e.g., `https://ai-music-gen.vercel.app`)
- Add to your resume/portfolio
- Share on social media
- Add to your GitHub README

### Monitor Performance
- Check Vercel Analytics dashboard
- Watch for errors in Vercel logs
- Monitor Core Web Vitals

### Iterate
- Push updates to GitHub → Auto-deploys
- Use preview deployments for testing
- Roll back if needed (Vercel keeps history)

## 🎓 Next Steps

1. **Add Features**
   - User accounts
   - Save/share compositions
   - More instruments
   - Actual trained model weights

2. **Improve Performance**
   - Lazy load TensorFlow.js
   - Add service worker for offline support
   - Optimize bundle size

3. **Scale**
   - Add backend API (Next.js API routes)
   - Database for user data (Vercel Postgres)
   - Authentication (NextAuth.js)

---

**Questions?** Check:
- [Vercel Docs](https://vercel.com/docs)
- [Next.js Deployment](https://nextjs.org/docs/app/building-your-application/deploying)
- [Vercel Support](https://vercel.com/support)

**Happy Deploying! 🎉**
