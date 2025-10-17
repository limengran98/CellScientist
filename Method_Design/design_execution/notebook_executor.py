from nbclient import NotebookClient
import nbformat
from typing import Optional

def execute_notebook(in_nb: str, out_nb: Optional[str] = None, timeout: int = 1200) -> str:
    nb = nbformat.read(in_nb, as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3", allow_errors=True)
    client.execute()
    out = out_nb or in_nb.replace(".ipynb", "_exec.ipynb")
    nbformat.write(nb, out)
    return out
