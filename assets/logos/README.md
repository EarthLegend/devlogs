# Logo Assets

This directory holds brand logos for LGND AI and Clay used in the site header and footer.

## Required Files

| File | Usage | Recommended Format |
|------|-------|--------------------|
| `lgnd-ai-logo.svg` | Navbar (28 px tall) and footer (24 px tall) | SVG preferred; PNG fallback acceptable |
| `clay-logo.svg` | Footer (20 px tall) | SVG preferred; PNG fallback acceptable |

## Usage Guidelines

- **SVG is strongly preferred** — scales crisply at any DPI and keeps the page weight low.
- If only PNG is available, use `@2×` assets (e.g. 56 px tall for the navbar logo) so they appear sharp on high-DPI displays.
- Keep file sizes small (< 20 KB for SVG, < 50 KB for PNG).

## Dark Mode

The site supports `prefers-color-scheme: dark`. Logos should be legible on both light (`#ffffff`) and dark (`#0f172a`) backgrounds. Options:

1. **Single adaptive SVG** — use `currentColor` fills so the logo inherits the surrounding text color.
2. **Two separate assets** — name them `lgnd-ai-logo-light.svg` and `lgnd-ai-logo-dark.svg` and update `_header.html` / `_footer.html` accordingly.

## Placeholder

Until the real logos are available, the header and footer gracefully fall back to text-only branding via `onerror` handlers and progressive-enhancement JavaScript.

## File Naming Convention

```
assets/
└── logos/
    ├── lgnd-ai-logo.svg   ← LGND AI primary logo
    ├── clay-logo.svg      ← Clay primary logo
    └── README.md          ← this file
```
