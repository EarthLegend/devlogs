# Devlogs — LGND AI

Research notes and devlogs from [LGND AI](https://lgnd.ai/) and [Clay](https://www.madewithclay.org/) — open geospatial AI research on foundation models, embeddings, and remote sensing.

Live at: **[devlogs.lgnd.ai](https://devlogs.lgnd.ai/)**

## Contributing

Posts live under `posts/` as executed Jupyter notebooks. To add a new post:

1. Push to `main` — GitHub Actions renders and deploys automatically.

<details>
<summary>Post template</summary>

Create a folder:

```
posts/YYYY-MM-DD-your-post-slug/
  index.ipynb    # the post (Jupyter notebook)
  assets/        # figures, images
```

Add Quarto front-matter in the first cell:

```yaml
---
title: "Your Post Title"
date: YYYY-MM-DD
author: Your Name
categories: [geospatial, embeddings]
description: "One or two sentences shown in the listing card."
image: assets/cover.png
---
```

Use `#| echo: false` to hide code cells from readers. Code output is cached (`freeze: auto`).

</details>

## Local development

```bash
pip install -r requirements.txt
quarto preview
```

Requires [Quarto CLI](https://quarto.org/docs/get-started/) >= 1.4.

## Built with

[Quarto](https://quarto.org) + [Jupyter](https://jupyter.org) + [GitHub Pages](https://pages.github.com)

## License

Posts: [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Code: [MIT](LICENSE)
