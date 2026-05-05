#!/usr/bin/env python3
"""
Convert Jupyter notebook (.ipynb) to Python file (.py)
Markdown cells become comment blocks, code cells remain as code.
"""

import json
import sys
from pathlib import Path


def notebook_to_python(notebook_path, output_path=None):
    """
    Convert a Jupyter notebook to a Python file.
    
    Args:
        notebook_path: Path to the .ipynb file
        output_path: Path to write the .py file (default: same name with .py extension)
    
    Returns:
        Path to the created Python file
    """
    notebook_path = Path(notebook_path)
    
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")
    
    if not notebook_path.suffix == '.ipynb':
        raise ValueError(f"File must be a .ipynb file: {notebook_path}")
    
    # Set output path
    if output_path is None:
        output_path = notebook_path.with_suffix('.py')
    else:
        output_path = Path(output_path)
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Extract cells and convert
    py_lines = []
    cells = notebook.get('cells', [])
    
    for cell in cells:
        cell_type = cell.get('cell_type')
        source = cell.get('source', [])
        
        # Join source lines into a single string
        if isinstance(source, list):
            source_text = ''.join(source)
        else:
            source_text = source
        
        if cell_type == 'markdown':
            # Convert markdown to comment block
            if source_text.strip():
                py_lines.append('# ' + '\n# '.join(source_text.strip().split('\n')))
                py_lines.append('')
        
        elif cell_type == 'code':
            # Keep code as is, skip empty cells
            if source_text.strip():
                py_lines.append(source_text)
                py_lines.append('')
    
    # Write to Python file
    py_content = '\n'.join(py_lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(py_content)
    
    return output_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python notebook_to_py.py <notebook.ipynb> [output.py]")
        sys.exit(1)
    
    notebook_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        result = notebook_to_python(notebook_path, output_path)
        print(f"✓ Converted: {notebook_path} → {result}")
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
