# CSE-450-Machine-learning

My work for the CSE 450 class.

## Tools

This repository includes three command-line tools:

1. `Tools/extractCSV.py` downloads a CSV from a URL and saves it locally with the same filename.
2. `compareCSV.py` compares two CSV files, reports how similar they are, and then walks you through a terminal menu for derived CSV outputs.
3. `notebook_to_py.py` converts a Jupyter notebook into a plain Python file.

Run the commands from the repository root. If you are using the bundled virtual environment, replace `python` with `./.venv/bin/python`.

## `Tools/extractCSV.py`

Downloads a web-based CSV file and saves it to a local directory using the same filename as the source URL.

### Usage

```bash
./.venv/bin/python Tools/extractCSV.py <csv-url> [-d DESTINATION]
```

### Arguments

`<csv-url>`
: HTTP or HTTPS URL to the CSV file.

`-d`, `--destination`
: Directory where the file should be saved. The directory is created if it does not already exist. Defaults to the current directory.

### Examples

Save a file in the current directory:

```bash
./.venv/bin/python Tools/extractCSV.py https://example.com/data/sample.csv
```

Save a file into a separate folder:

```bash
./.venv/bin/python Tools/extractCSV.py https://example.com/data/sample.csv --destination downloads
```

### Behavior

- The output filename is taken from the last path segment of the URL.
- The file is written exactly once with the same name as the remote file.
- If the URL path does not contain a filename, the script raises an error.

## `compareCSV.py`

Compares two CSV files and prints how much data they share in both directions. After the report, it opens a terminal menu so you can choose whether to write derived CSVs for the sum, differences, intersection, and XOR of the rows.

The script is tolerant of small differences:

- headers are compared case-insensitively after trimming whitespace
- text values are compared case-insensitively unless you opt out
- numeric values can be compared with a tolerance
- rows are matched by normalized values instead of raw text
- duplicate rows are handled correctly

### Usage

```bash
./.venv/bin/python compareCSV.py <csv-a> <csv-b>
```

### Arguments

`<csv-a>`
: First CSV file.

`<csv-b>`
: Second CSV file.

### Menu flow

After you start the command, the script prompts for:

1. numeric tolerance for comparisons
2. whether text should be compared case-sensitively
3. the delimiter to use for any generated output files
4. which output files to create
5. the output path for each selected file

The menu choices are:

1. sum / union of both files
2. rows in the first file that are not in the second
3. rows in the second file that are not in the first
4. rows common to both files
5. rows that appear in only one file
6. create every output
7. finish without writing more files

### Output report

The script always prints a two-line similarity summary unless it cannot compare the files.

Example output:

```text
file1.csv shares 100.00% of its data with file2.csv (25/25 rows)
file2.csv shares 96.15% of its data with file1.csv (25/26 rows)
```

If the files do not use identical headers, the script also prints a header-alignment note showing how many shared columns were compared.

### Examples

Compare two CSV files and exit after the similarity report:

```bash
./.venv/bin/python compareCSV.py data/a.csv data/b.csv
```

Create the common-row output from the menu after the report appears:

```bash
./.venv/bin/python compareCSV.py data/a.csv data/b.csv
```

Create every derived output by choosing menu option `6`:

```bash
./.venv/bin/python Tools/compareCSV.py data/a.csv data/b.csv
```

### Behavior

- Rows are matched using normalized header names, so `Name` and `name` are treated as the same column.
- Numeric values such as `1`, `1.0`, and `1.00` are treated as equivalent unless you set a tighter tolerance.
- The generated CSVs keep the column order from the two input files, combined without duplicates.
- If you request no output files, the script only prints the similarity summary.
- The default output filenames use the two input file stems, but you can replace them when the menu asks for a path.

## `notebook_to_py.py`

Converts a Jupyter notebook into a `.py` file. Markdown cells become comment blocks and code cells stay as code.

### Usage

```bash
./.venv/bin/python notebook_to_py.py <notebook.ipynb> [output.py]
```

### Arguments

`<notebook.ipynb>`
: Input notebook file.

`[output.py]`
: Optional destination Python file. If omitted, the script writes a `.py` file next to the notebook with the same base name.

### Examples

Convert a notebook in place to a Python file with the same base name:

```bash
./.venv/bin/python notebook_to_py.py notebooks/Exploration_01.ipynb
```

Convert a notebook and choose a custom output path:

```bash
./.venv/bin/python notebook_to_py.py notebooks/Exploration_01.ipynb notebooks/Exploration_01_export.py
```

### Behavior

- The input file must end in `.ipynb`.
- The script fails if the notebook does not exist.
- Markdown text is converted into `#` comment lines in the output file.
- Empty notebook cells are skipped.
