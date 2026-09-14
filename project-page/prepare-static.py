"""Populate the default accessible chart from the manuscript data (no build dependencies)."""
from pathlib import Path
import json, re, html
root = Path(__file__).resolve().parent
data = json.loads((root/'dist/assets/results.json').read_text(encoding='utf-8'))
rows = [r for r in data['rows'] if r['budget'] == 5]
bars = []
table = []
for row in rows:
    group = f'<div class="chart-group"><div class="chart-group-title">{row["dataset"]} <span>{row["split"]} split</span></div>'
    for key in ['full','random']:
        value = row['results'][key]
        mean, std = value['mean'], value['std']
        lower, upper = max(0,mean-std)/.4*100, min(.4,mean+std)/.4*100
        name = 'CellScientist' if key=='full' else 'Random routing'
        group += f'<div class="bar-row {key}" title="{name}: {mean:.4f} ± {std:.4f}"><div class="bar-track"><span class="bar-fill" style="--value:{mean/.4*100}%"></span><span class="bar-whisker" style="--lower:{lower}%;--range:{upper-lower}%"></span></div><span class="bar-value">{mean:.4f}</span></div>'
    bars.append(group+'</div>')
    table.append(f'<tr><th scope="row">{row["dataset"]} / {row["split"]}</th>'+''.join(f'<td>{html.escape(row["results"][key]["display"])}</td>' for key in ['fixed','flat','random','aide','full'])+'</tr>')
path=root/'dist/index.html'
text=path.read_text(encoding='utf-8')
start=text.index('<div id="budget-chart"')
end=text.index('<div class="chart-scale">',start)
opening=text[start:text.index('>',start)+1]
text=text[:start]+opening+''.join(bars)+'</div>'+text[end:]
text=re.sub(r'(<tbody id="results-table-body">).*?(</tbody>)',lambda m:m[1]+''.join(table)+m[2],text,flags=re.S)
path.write_text(text,encoding='utf-8')
print('Default chart and accessible table populated from manuscript Table 3.')
