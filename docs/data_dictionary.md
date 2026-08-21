# Data Dictionary

## Source
Official FHWA National Bridge Inventory (NBI) comma-delimited files for Maine (State Code 23).

## File Naming Convention
- `ME23.txt` — 2023 data
- `ME24.txt` — 2024 data
- `ME25.txt` — 2025 data

## Record Format
Each file contains one header row followed by one bridge record per row. Fields are comma-separated with single-quote text qualifiers.

## Key Field Definitions

### Bridge Identifier
| Field Name | NBI Item | Description |
|-----------|----------|-------------|
| `STATE_CODE_001` | 1 | State code (23 = Maine) |
| `STRUCTURE_NUMBER_008` | 8 | Unique bridge identifier within the state |

Combined bridge ID: `STATE_CODE_001 + "_" + STRUCTURE_NUMBER_008`

### Condition Ratings (0–9 scale)
| Field Name | NBI Item | Description |
|-----------|----------|-------------|
| `DECK_COND_058` | 58 | Deck condition rating |
| `SUPERSTRUCTURE_COND_059` | 59 | Superstructure condition rating |
| `SUBSTRUCTURE_COND_060` | 60 | Substructure condition rating |
| `CULVERT_COND_062` | 62 | Culvert condition rating (if applicable) |

**Condition classes:**
- **Good:** 7, 8, 9
- **Fair:** 5, 6
- **Poor:** 0, 1, 2, 3, 4

Missing or not applicable values are coded as `N` in the raw files and converted to `NaN` during cleaning.

### Traffic Features
| Field Name | NBI Item | Description |
|-----------|----------|-------------|
| `ADT_029` | 29 | Average Daily Traffic (vehicles per day) |
| `PERCENT_ADT_TRUCK_109` | 109 | Percentage of ADT that is truck traffic |

### Structural Features
| Field Name | NBI Item | Description |
|-----------|----------|-------------|
| `YEAR_BUILT_027` | 27 | Year the structure was built |
| `MAIN_UNIT_SPANS_045` | 45 | Number of spans in the main unit |
| `MAX_SPAN_LEN_MT_048` | 48 | Length of maximum span (meters) |
| `STRUCTURE_LEN_MT_049` | 49 | Structure length (meters) |
| `ROADWAY_WIDTH_MT_051` | 51 | Bridge roadway width, curb-to-curb (meters) |
| `DECK_WIDTH_MT_052` | 52 | Deck width, out-to-out (meters) |
| `TRAFFIC_LANES_ON_028A` | 28A | Number of traffic lanes on the structure |

### Material and Type
| Field Name | NBI Item | Description |
|-----------|----------|-------------|
| `STRUCTURE_KIND_043A` | 43A | Kind of material/design (e.g., concrete, steel) |
| `STRUCTURE_TYPE_043B` | 43B | Type of design/construction (e.g., stringer/multi-beam) |
| `DECK_STRUCTURE_TYPE_107` | 107 | Deck structure type |
| `SURFACE_TYPE_108A` | 108A | Type of wearing surface |
| `DECK_PROTECTION_108C` | 108C | Deck protection system |

### Risk and Operational Features
| Field Name | NBI Item | Description |
|-----------|----------|-------------|
| `SCOUR_CRITICAL_113` | 113 | Scour critical bridge status |
| `WATERWAY_EVAL_071` | 71 | Waterway adequacy evaluation |
| `FUNCTIONAL_CLASS_026` | 26 | Functional class of inventory route |
| `HIGHWAY_SYSTEM_104` | 104 | Highway system of inventory route |
| `OPEN_CLOSED_POSTED_041` | 41 | Structure open/posted/closed status |

### Computed Features
| Field Name | Description |
|-----------|-------------|
| `bridge_age` | `inspection_year - YEAR_BUILT_027` |
| `inspection_year` | Year of inspection (2023, 2024, or 2025) |

## Missing Value Codes
In the raw files, missing or not-applicable values are represented by:
- `N` — Not applicable or missing
- Blank/empty string — Missing

These are converted to `NaN` during data preparation.

## Derived Target
| Field Name | Description |
|-----------|-------------|
| `target_deck_poor_next_year` | 1 if next-year `DECK_COND_058` ≤ 4, else 0 |
