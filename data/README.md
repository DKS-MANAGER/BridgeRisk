# Raw Data Download Instructions

**Author:** Divyansh Kumar Singh (DKS) · M.Tech Civil Engineering (Hydraulic), IIT Kanpur  
**GitHub:** [DKS-MANAGER](https://github.com/DKS-MANAGER)

## Data Source
Official Federal Highway Administration (FHWA) National Bridge Inventory (NBI) ASCII files:
`https://www.fhwa.dot.gov/bridge/nbi/ascii.cfm`

## Selected States (2021–2025)
- **Maine (State Code: 23):** Northeast freeze-thaw cycles, winter road salt, coastal exposure.
- **Hawaii (State Code: 15):** High marine humidity, tropical rain, airborne chloride corrosion.
- **Delaware (State Code: 10):** High-density mid-Atlantic freight and commuter corridor.

## How to Download
To download raw archive files automatically from FHWA:
```bash
python src/download_data.py
```

The script retrieves:
- 2021–2024: FHWA national ZIP archives (`{year}del.zip`) and extracts state records to `data/raw/extracted/{year}/`.
- 2025: State delimited files (`{state}25.txt`) to `data/raw/extracted/2025/`.

## Preprocessed Data
For immediate modeling without re-downloading raw archives, the repository includes pre-cleaned datasets:
- `data/processed/train_2021_2024.parquet`: 13,618 consecutive annual transition pairs (2021→2022, 2022→2023, 2023→2024).
- `data/processed/test_2024_2025.parquet`: 4,558 bridges for out-of-time evaluation (features from 2024, target from 2025).
