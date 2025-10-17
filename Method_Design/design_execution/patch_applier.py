import os, json, re
import nbformat
def _find_anchor_idx(nb, anchor):
    try: return int(anchor)
    except Exception: pass
    for i, c in enumerate(nb.cells):
        src = (c.source or "")
        if anchor and isinstance(anchor, str) and anchor.strip():
            if c.cell_type == "markdown" and anchor in src: return i
            if c.cell_type == "code" and anchor in src: return i
    return None
def apply_patch(in_nb_path: str, patch_json_path: str) -> str:
    nb = nbformat.read(in_nb_path, as_version=4)
    patch = json.loads(open(patch_json_path, "r", encoding="utf-8").read())
    cells_to_add = patch.get("cells_to_add") or []
    cells_to_replace = patch.get("cells_to_replace") or []
    for item in cells_to_replace:
        anchor = item.get("anchor"); content = item.get("content", ""); ctype = item.get("cell_type", "code")
        idx = _find_anchor_idx(nb, anchor); 
        if idx is None: continue
        new_cell = nbformat.v4.new_code_cell(content) if ctype == "code" else nbformat.v4.new_markdown_cell(content)
        nb.cells[idx] = new_cell
    for item in cells_to_add:
        position = item.get("position", "end"); anchor = item.get("anchor"); content = item.get("content", ""); ctype = item.get("cell_type", "code")
        new_cell = nbformat.v4.new_code_cell(content) if ctype == "code" else nbformat.v4.new_markdown_cell(content)
        if position == "start":
            nb.cells.insert(0, new_cell)
        elif position in ("after", "before"):
            idx = _find_anchor_idx(nb, anchor)
            if idx is None: nb.cells.append(new_cell)
            else:
                insert_at = idx + 1 if position == "after" else idx
                nb.cells.insert(insert_at, new_cell)
        else:
            nb.cells.append(new_cell)
    out_path = in_nb_path.replace(".ipynb", "_patched.ipynb")
    nbformat.write(nb, out_path); return out_path
