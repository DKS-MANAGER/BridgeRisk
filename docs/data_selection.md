# Data Selection Documentation

## Selected State
**Maine (State Code: 23)**

## Bridge Counts (from FHWA official pages)
| Year | Number of Highway Bridges |
|------|---------------------------|
| 2023 | 2,521 |
| 2024 | 2,518 |
| 2025 | 2,542 |

## Files Downloaded
| Year | Source URL | File Name | Format |
|------|-----------|-----------|--------|
| 2023 | https://www.fhwa.dot.gov/bridge/nbi/2023del.zip | ME23.txt | Comma-delimited |
| 2024 | https://www.fhwa.dot.gov/bridge/nbi/2024del.zip | ME24.txt | Comma-delimited |
| 2025 | https://www.fhwa.dot.gov/bridge/nbi/2025/delimited/ME25.txt | ME25.txt | Comma-delimited |

## Selection Rationale
1. Maine has approximately 2,500 bridges per year, which falls within the target range of 2,000–3,000.
2. The bridge identifier (STATE_CODE_001 + STRUCTURE_NUMBER_008) is expected to be consistent across years.
3. Condition-rating fields (DECK_COND_058, SUPERSTRUCTURE_COND_059, SUBSTRUCTURE_COND_060) are expected to be present and not mostly missing.
4. Maine is a single state, avoiding the need to combine multiple states.

## Notes
- The 2023 and 2024 data were obtained from the FHWA "all states" delimited ZIP files and filtered to Maine.
- The 2025 data was obtained directly as a delimited text file.
- All files are comma-delimited with a header row.
