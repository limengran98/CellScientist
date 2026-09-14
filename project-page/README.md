# CellScientist project page

[Visit the project page](https://cellscientist-research.phuonganh49123.chatgpt.site) · [Read the paper](https://cellscientist-research.phuonganh49123.chatgpt.site/assets/CellScientist.pdf) · [Download the data](https://huggingface.co/datasets/Boom5426/CellScientist)

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

The public page is hosted on Sites. This directory contains the corresponding static source; publish an updated `dist/` through the existing Site after reviewing changes. GitHub commits and Site publication are separate release steps. Use relative asset paths so local previews and hosted copies render consistently.
