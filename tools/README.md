# Tools

Utility scripts used across modules for data handling and notebook management.

| Script | Description |
|--------|-------------|
| `compareCSV.py` | Compare two CSV files with row-wise set operations (intersection, difference, etc.) |
| `extractCSV.py` | Download a CSV from a web URL and save it locally |
| `notebook_to_py.py` | Convert Jupyter notebooks (`.ipynb`) to plain Python scripts |

## Usage

All scripts are standalone and can be run from the command line:

```bash
python compareCSV.py <file1.csv> <file2.csv>
python extractCSV.py <url>
python notebook_to_py.py <notebook.ipynb>
```
