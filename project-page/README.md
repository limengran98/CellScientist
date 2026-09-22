# CellScientist project page

[Visit the project page](https://limengran98.github.io/CellScientist) · [Read the paper](https://limengran98.github.io/CellScientist/assets/CellScientist.pdf) · [Download the data](https://huggingface.co/datasets/Boom5426/CellScientist)

Source for the research page accompanying **CellScientist: Model Revision by Diagnostic Routing for Morphological Perturbation Prediction**. It includes seven manuscript figures, interactive budget comparisons, component audits, author affiliations and the named-author preprint.

## Local preview

From the repository root:

```bash
python -m http.server 4173 --bind 127.0.0.1 --directory project-page/dist
```

Open `http://127.0.0.1:4173`. The page uses static HTML, CSS and JavaScript; all figure and paper assets are included.

## Editing the page

| File | Purpose |
| --- | --- |
| `dist/index.html` | Research narrative, figures, authors, links and citation |
| `dist/styles.css` | Layout and responsive styling |
| `dist/app.js` | Budget selector, dataset selector, figure viewer and copy buttons |
| `dist/assets/results.json` | Numeric data for the interactive comparison |
| `dist/assets/CellScientist.pdf` | Named-author preprint |
| `prepare-static.py` | Regenerate the default chart and table after changing results |

To regenerate the default accessible chart:

```bash
python project-page/prepare-static.py
```

The chart reports five-seed held-out global PCC from manuscript Table 3. Per-policy whiskers are sample standard deviations; the aggregate CellScientist-minus-Random effect uses a separate paired-setting 95% t-interval. The BBBC047 open-workflow trajectory reports validation scores used during model search. Keep these evaluation roles and the PDF consistent when updating results.

## GitHub Pages deployment

The public project page is hosted at https://limengran98.github.io/CellScientist/.
Changes pushed to `main` under `project-page/` are automatically published by
[`Deploy project page`](../.github/workflows/project-page.yml). The workflow
uploads `dist/` directly; no package installation or build service is needed.
It can also be run manually from the repository's Actions tab.

Keep asset and data paths relative so local previews and the repository URL
work identically. The paper download is `dist/assets/CellScientist.pdf`.
