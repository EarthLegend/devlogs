# Devlogs — LGND AI

A science and technology blog by LGND AI. Technical posts on geospatial AI,
embeddings, and foundation models.

Live at: **[earthlegend.github.io/devlogs](https://earthlegend.github.io/devlogs/)**

Built with [Quarto](https://quarto.org). Posts are Jupyter notebooks. Deploys automatically to GitHub Pages on every push to `main`.

---

## Adding a new post

1. Create a folder under `posts/`:
   ```
   posts/YYYY-MM-DD-your-post-slug/
   ├── index.ipynb    # the post
   └── assets/        # figures, images
   ```

2. Add Quarto front-matter in the first Markdown cell of the notebook:
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

3. Write the notebook. Code cells run and are cached (`freeze: auto`). Use `#| echo: false` to hide cells from readers.

4. Push to `main`. GitHub Actions renders and deploys automatically.

---

## Local development

```bash
pip install -r requirements.txt
quarto preview
```

Requires [Quarto CLI](https://quarto.org/docs/get-started/) ≥ 1.4.

---

## License

Posts: [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) · Code: [MIT](LICENSE)
