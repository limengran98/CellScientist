'use strict';
const dialog = document.querySelector('#figure-dialog');
document.querySelectorAll('[data-zoom]').forEach(button => button.addEventListener('click', () => {
  const image = dialog.querySelector('img');
  image.src = button.dataset.zoom;
  image.alt = button.querySelector('img').alt;
  document.querySelector('#figure-caption').textContent = button.dataset.caption;
  dialog.showModal();
}));
document.querySelector('#close-figure').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', e => { if (e.target === dialog) { const b = dialog.getBoundingClientRect(); if(e.clientX < b.left || e.clientX > b.right || e.clientY < b.top || e.clientY > b.bottom) dialog.close(); } });

const escapeHTML = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let resultsData;
let activeBudget = 5;
const number = n => n.toFixed(4).replace('-', '−');
function barsFor(rows) {
  return rows.map(row => `<div class="chart-group"><div class="chart-group-title">${escapeHTML(row.dataset)} <span>${escapeHTML(row.split)} split</span></div>${['full','random'].map(key => {
    const value = row.results[key];
    const lower = Math.max(0, value.mean - value.std) / .4 * 100;
    const upper = Math.min(.4, value.mean + value.std) / .4 * 100;
    return `<div class="bar-row ${key}" title="${key === 'full' ? 'CellScientist' : 'Random routing'}: ${number(value.mean)} ± ${number(value.std)}"><div class="bar-track"><span class="bar-fill" style="--value:${value.mean / .4 * 100}%"></span><span class="bar-whisker" style="--lower:${lower}%;--range:${upper-lower}%"></span></div><span class="bar-value">${number(value.mean)}</span></div>`;
  }).join('')}</div>`).join('');
}
function selectBudget(budget, announce = true) {
  activeBudget = budget;
  document.querySelectorAll('[data-budget]').forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.budget) === budget)));
  const rows = resultsData.rows.filter(row => row.budget === budget);
  const effect = resultsData.effects.rows.find(row => row.budget === budget);
  const chart = document.querySelector('#budget-chart');
  chart.innerHTML = barsFor(rows);
  chart.setAttribute('aria-label', `Held-out global PCC at ${budget} candidate evaluations. ${rows.map(r=>`${r.dataset} ${r.split}: CellScientist ${number(r.results.full.mean)}, Random routing ${number(r.results.random.mean)}`).join('. ')}`);
  document.querySelector('#effect-value').textContent = `+${number(effect.mean)}`;
  document.querySelector('#effect-ci').textContent = `[${number(effect.lower)}, ${number(effect.upper)}]`;
  document.querySelector('#effect-note').textContent = budget === 10 ? 'At ten evaluations, the mean gap narrows. The interval includes zero, so the evidence is strongest for finding better candidates earlier.' : 'The positive confidence interval supports a mean advantage at this small evaluation budget, where useful revision decisions matter early.';
  document.querySelector('#results-table-body').innerHTML = rows.map(row=>`<tr><th scope="row">${escapeHTML(row.dataset)} / ${escapeHTML(row.split)}</th>${['fixed','flat','random','aide','full'].map(key=>`<td>${escapeHTML(row.results[key].display)}</td>`).join('')}</tr>`).join('');
  document.querySelector('#table-budget').textContent = `Budget: ${budget} evaluations.`;
  if(announce) document.querySelector('#chart-status').textContent = `Showing ${budget} evaluations. Mean CellScientist advantage: ${number(effect.mean)} PCC. 95% confidence interval ${number(effect.lower)} to ${number(effect.upper)}.`;
}
document.querySelectorAll('[data-budget]').forEach(button => button.addEventListener('click', () => { if(resultsData) selectBudget(Number(button.dataset.budget)); }));
fetch('assets/results.json').then(response => { if(!response.ok) throw new Error('Chart data unavailable'); return response.json(); }).then(data => {resultsData = data; selectBudget(activeBudget, false);}).catch(() => {
  document.querySelector('#chart-status').textContent = 'Interactive data could not load. The default comparison and downloadable data remain available.';
  document.querySelectorAll('[data-budget]').forEach(button => button.disabled = true);
});

document.querySelectorAll('[data-dataset]').forEach(button => button.addEventListener('click', () => {
  const dataset = button.dataset.dataset;
  document.querySelectorAll('[data-dataset]').forEach(b=>b.setAttribute('aria-pressed', String(b === button)));
  document.querySelectorAll('.dataset-label').forEach(label=>label.textContent = dataset);
  const distribution = document.querySelector('#distribution-image');
  const trajectory = document.querySelector('#trajectory-image');
  distribution.src = `assets/Figure3_${dataset}.webp`;
  trajectory.src = `assets/Figure4_${dataset}.webp`;
  distribution.alt = `${dataset} five-fold R-squared distributions comparing CS-model, RealMLP, RF-TD, and TabR.`;
  trajectory.alt = `${dataset} candidate validation PCC and best-so-far scores across ten model-revision iterations.`;
  const dZoom = document.querySelector('#distribution-zoom');
  const tZoom = document.querySelector('#trajectory-zoom');
  dZoom.dataset.zoom = distribution.getAttribute('src');
  tZoom.dataset.zoom = trajectory.getAttribute('src');
  dZoom.dataset.caption = `${dataset}: five-fold R² distributions for CS-model and fixed baselines. Statistical annotations are reproduced from the manuscript.`;
  tZoom.dataset.caption = `${dataset}: candidate validation PCC and retained best-so-far scores during open-workflow model revision.`;
}));

document.querySelectorAll('[data-copy]').forEach(button => button.addEventListener('click', async () => {
  const code = document.getElementById(button.dataset.copy);
  const original = button.textContent;
  try {
    await navigator.clipboard.writeText(code.textContent);
    button.textContent = 'Copied ✓';
    document.querySelector('#copy-status').textContent = button.dataset.copy === 'citation' ? 'Citation copied to clipboard.' : 'Quick-start commands copied to clipboard.';
    setTimeout(()=>button.textContent = original,2200);
  } catch {
    const range = document.createRange(); range.selectNodeContents(code);
    const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
    document.querySelector('#copy-status').textContent = 'Text selected. Press Ctrl+C or Command+C to copy.';
  }
}));
