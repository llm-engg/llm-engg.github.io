# Analytics Setup Guide

This guide explains how to add analytics to your LLM Engineering Course website.

## Google Analytics 4 (Recommended)

### Step 1: Create Google Analytics Property

1. Go to [Google Analytics](https://analytics.google.com)
2. Sign in with your Google account
3. Click "Start measuring"
4. Create an account name (e.g., "LLM Engineering Course")
5. Choose "Web" as the platform
6. Enter your website details:
   - Website name: LLM Engineering Course
   - Website URL: https://llm-engg.github.io
   - Industry category: Education
   - Reporting time zone: Your timezone
7. Accept the terms and create the property
8. Copy your **Measurement ID** (format: G-XXXXXXXXXX)

### Step 2: Update Configuration

Replace `G-LLD65KQHZH` in `mkdocs.yml` with your actual Measurement ID:

```yaml
extra:
  analytics:
    provider: google
    property: G-1234567890  # Your actual ID here
```

### Step 3: Deploy Changes

```bash
mkdocs gh-deploy
```

## Alternative Analytics Options

### 1. Plausible Analytics (Privacy-focused)

```yaml
extra:
  analytics:
    provider: custom
    property: plausible
  custom_analytics: |
    <script defer data-domain="llm-engg.github.io" src="https://plausible.io/js/script.js"></script>
```

### 2. Matomo (Self-hosted)

```yaml
extra:
  analytics:
    provider: custom
  custom_analytics: |
    <script>
      var _paq = window._paq = window._paq || [];
      _paq.push(['trackPageView']);
      _paq.push(['enableLinkTracking']);
      (function() {
        var u="//your-matomo-domain.com/";
        _paq.push(['setTrackerUrl', u+'matomo.php']);
        _paq.push(['setSiteId', '1']);
        var d=document, g=d.createElement('script'), s=d.getElementsByTagName('script')[0];
        g.type='text/javascript'; g.async=true; g.defer=true; g.src=u+'matomo.js'; s.parentNode.insertBefore(g,s);
      })();
    </script>
```

### 3. Simple Analytics

```yaml
extra:
  analytics:
    provider: custom
  custom_analytics: |
    <script async defer src="https://scripts.simpleanalyticscdn.com/latest.js"></script>
    <noscript><img src="https://queue.simpleanalyticscdn.com/noscript.gif" alt="" referrerpolicy="no-referrer-when-downgrade" /></noscript>
```

## What Analytics Will Track

With Google Analytics 4, you'll get insights on:

- **Page Views**: Most popular course sections
- **User Behavior**: Time spent on pages, bounce rate
- **Traffic Sources**: How students find your course
- **Demographics**: Geographic distribution of students
- **Device Usage**: Desktop vs mobile usage
- **Real-time Data**: Current active users

## Feedback Integration

The current configuration also includes feedback widgets that allow students to rate page helpfulness. This provides qualitative data alongside quantitative analytics.

## Privacy Considerations

- Consider adding a privacy policy mentioning analytics
- Google Analytics 4 is more privacy-focused than Universal Analytics
- Alternative options like Plausible offer cookie-free tracking
- You can disable analytics in development by setting `MKDOCS_CONFIG` environment variable

## Testing Analytics

After deployment:

1. Visit your site
2. Check Google Analytics Real-Time reports
3. Verify page views are being tracked
4. Test the feedback widgets if enabled

## Advanced Features

### Custom Events

Track specific interactions like:
- Assignment downloads
- External link clicks
- Video plays (if you add videos later)

### Goal Setting

Set up goals for:
- Course completion tracking
- Time spent on course
- Assignment submission rates

---

Replace `G-XXXXXXXXXX` in your `mkdocs.yml` with your actual Google Analytics Measurement ID to start tracking!