# CSE-450-Machine-learning

My work for the CSE 450 class.

## Usage

`extractCSV.py` downloads a CSV from a web URL and saves it locally using the same filename.

### Run it

```bash
python extractCSV.py <csv-url>
```

By default, the file is saved in the current folder. You can also choose a destination directory:

```bash
python extractCSV.py <csv-url> --destination downloads
```

### Example

```bash
python extractCSV.py https://example.com/data/sample.csv
```

That command saves the file as `sample.csv` in the current directory.
