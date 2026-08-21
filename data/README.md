# Raw Data Download Instructions

**Author:** Divyansh Kumar Singh (DKS) · M.Tech Civil Engineering (Hydraulic), IIT Kanpur  
**GitHub:** [DKS-MANAGER](https://github.com/DKS-MANAGER)

## Data Source
Official FHWA National Bridge Inventory (NBI) ASCII files.

## Selected State
Maine (State Code: 23)

## Files to Download
- 2023: `https://www.fhwa.dot.gov/bridge/nbi/2023/ME25.txt` (no delimiter, fixed-width)
- 2024: `https://www.fhwa.dot.gov/bridge/nbi/2024/ME25.txt` (no delimiter, fixed-width)
- 2025: `https://www.fhwa.dot.gov/bridge/nbi/2025/delimited/ME25.txt` (comma-delimited)

## How to Download
Run the automated download script:
```bash
python src/download_data.py
```

Or download manually and place files in `data/raw/downloads/`.

## Notes
- The 2023 and 2024 files are fixed-width format (445 characters per record).
- The 2025 file is comma-delimited with a header row.
- See `docs/data_selection.md` for the rationale behind selecting Maine.
