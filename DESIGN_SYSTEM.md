# LGND AI Devlogs — Design System

A concise reference for contributors and designers working on the devlogs website.

---

## Colour Palette

### Light Mode

| Token | Hex | Usage |
|-------|-----|-------|
| `--color-primary` | `#0f2a4a` | Navbar background, headings |
| `--color-primary-light` | `#1a3f6f` | Hover states on primary surfaces |
| `--color-accent` | `#2563eb` | Links, active states, code border |
| `--color-accent-hover` | `#1d4ed8` | Link hover |
| `--color-bg` | `#ffffff` | Page background |
| `--color-bg-subtle` | `#f8fafc` | Section backgrounds, footer |
| `--color-bg-muted` | `#f1f5f9` | Code background, badges |
| `--color-border` | `#e2e8f0` | Dividers, card borders |
| `--color-text` | `#1e293b` | Body text |
| `--color-text-muted` | `#64748b` | Secondary text, captions |
| `--color-text-subtle` | `#94a3b8` | Meta text, dates |
| `--color-key-result-bg` | `#eff6ff` | Key-result callout background |

### Dark Mode

Activated automatically via `prefers-color-scheme: dark`. All tokens listed above have dark-mode overrides defined in `styles.css`.

| Notable change | Dark value |
|----------------|-----------|
| Page background | `#0f172a` |
| Accent (links) | `#60a5fa` |
| Primary (navbar) | `#1e3a5f` |

---

## Typography

### Font Families

| Role | Family | Weights |
|------|--------|---------|
| Body | Inter (Google Fonts) + system-ui fallback | 400, 500, 600, 700 |
| Headings (h1–h3) | Playfair Display (Google Fonts) + Georgia fallback | 600, 700 |
| Monospace | Fira Code (Google Fonts) + Cascadia Code fallback | 400, 500 |

### Font Scale

| Element | Size | Weight | Family |
|---------|------|--------|--------|
| `h1` | ~2.25 rem (Quarto default) | 700 | Playfair Display |
| `h2` | ~1.75 rem | 700 | Playfair Display |
| `h3` | ~1.4 rem | 700 | Playfair Display |
| `h4`–`h6` | Quarto default | 600 | Inter |
| Body | 17 px (16 px mobile, 18 px ≥ 1400 px) | 400 | Inter |
| Small / meta | 0.82–0.875 rem | 400–500 | Inter |
| Code | 0.875–0.88 em | 400 | Fira Code |

### Line Heights

| Context | Value |
|---------|-------|
| Body text | 1.75 |
| Headings | 1.25 |
| Captions / meta | 1.5 |

---

## Spacing

Uses an 8-point base grid with named tokens:

| Token | Value |
|-------|-------|
| `--space-1` | 0.25 rem (4 px) |
| `--space-2` | 0.5 rem (8 px) |
| `--space-3` | 0.75 rem (12 px) |
| `--space-4` | 1 rem (16 px) |
| `--space-6` | 1.5 rem (24 px) |
| `--space-8` | 2 rem (32 px) |
| `--space-12` | 3 rem (48 px) |
| `--space-16` | 4 rem (64 px) |

---

## Shadows

| Token | Value | Usage |
|-------|-------|-------|
| `--shadow-sm` | `0 1px 3px rgba(0,0,0,.08), 0 1px 2px rgba(0,0,0,.06)` | Cards at rest, code blocks |
| `--shadow-md` | `0 4px 6px rgba(0,0,0,.07), 0 2px 4px rgba(0,0,0,.06)` | Dropdowns |
| `--shadow-lg` | `0 10px 15px rgba(0,0,0,.08), 0 4px 6px rgba(0,0,0,.05)` | Card hover state |

---

## Border Radius

| Token | Value | Usage |
|-------|-------|-------|
| `--radius-sm` | 4 px | Badges, small elements |
| `--radius-md` | 8 px | Code blocks, callouts, card images |
| `--radius-lg` | 12 px | Post listing cards |

---

## Transitions

| Token | Value |
|-------|-------|
| `--transition-fast` | 150 ms ease |
| `--transition-normal` | 250 ms ease |

---

## Components

### Post Cards

Post listing cards use `.quarto-post`. Key behaviours:

- Border: `1px solid var(--color-border)`, `border-radius: var(--radius-lg)`
- Hover: elevates with `--shadow-lg`, translates up 2 px, accent border colour
- Thumbnail images use `object-fit: cover` with rounded top corners

### Callout Boxes

Standard Quarto callouts (`.callout`) gain:

- `border-radius: var(--radius-md)`
- `box-shadow: var(--shadow-sm)`

### Key Result Highlights

Apply class `key-result` to a `<div>` or `<p>`:

```html
<div class="key-result">
  The model achieves <strong>87.4% mAP</strong> on the held-out test set.
</div>
```

Renders with a left-border accent, blue tinted background, and subtle shadow.

### Code Blocks

`pre.sourceCode` gets:

- Left border in `--color-accent`
- Subtle background `--color-bg-subtle`
- Shadow `--shadow-sm`

Inline `code` gets a muted background pill.

### Navbar

- Background: `--color-primary` (`#0f2a4a`)
- Brand: bold, white, supports an optional logo image (`assets/logos/lgnd-ai-logo.svg`)
- Nav links: semi-transparent white, darken on hover with a background highlight

### Footer (custom `_footer.html`)

- Background: `--color-bg-subtle`
- 3-column responsive grid (Branding | Links | Resources)
- Dual logo placement: LGND AI (24 px) + Clay (20 px)
- Copyright bar with CC-BY and MIT license links

---

## Responsive Breakpoints

Follows Bootstrap 5 / Quarto defaults:

| Breakpoint | Width | Adjustments |
|------------|-------|-------------|
| Mobile | < 576 px | `font-size: 16 px`, reduced heading scale, tighter card padding |
| Tablet | 576–991 px | `font-size: 16.5 px` |
| Desktop | 992–1399 px | `font-size: 17 px` (default) |
| Wide | ≥ 1400 px | `font-size: 18 px` |

---

## Accessibility

- **Contrast**: all foreground/background pairings meet WCAG AA (≥ 4.5:1 for body text, ≥ 3:1 for large text).
- **Focus styles**: `a`, `button`, `input`, `select`, and `textarea` show a 2 px solid accent outline on `:focus-visible`.
- **Images**: logo images include `alt` text; decorative images should use `alt=""`.
- **Semantic HTML**: Quarto generates semantic landmarks; custom templates use `<footer>` correctly.
- **Reduced motion**: no CSS animations are used; transitions respect user preferences implicitly because Quarto relies on browser defaults.

---

## Dark Mode Implementation

Dark mode is applied via:

```css
@media (prefers-color-scheme: dark) {
  :root {
    /* overridden tokens */
  }
}
```

The approach:

1. All colours reference CSS variables — no hard-coded hex in component rules.
2. Dark token overrides are centralised in a single `@media` block at the top of `styles.css`.
3. Logo images fall back gracefully; for true dark-mode logos see `assets/logos/README.md`.

---

## Browser Support

Targeting the last 2 versions of evergreen browsers:

| Browser | Minimum version |
|---------|-----------------|
| Chrome / Edge | 109+ |
| Firefox | 115+ |
| Safari | 15.4+ |

CSS features used:

- CSS Custom Properties (variables) — universal support
- `@media (prefers-color-scheme)` — universal support
- CSS Grid (`auto-fit`) — universal support
- `:focus-visible` — universal support in target range
- `text-decoration-color` / `text-underline-offset` — universal support in target range
