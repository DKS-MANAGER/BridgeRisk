# FHWA NBI Data Dictionary & Field Reference

**Author:** Divyansh Kumar Singh (DKS) · M.Tech Civil Engineering (Hydraulic), IIT Kanpur  
**GitHub:** [DKS-MANAGER](https://github.com/DKS-MANAGER)

Reference guide for Federal Highway Administration (FHWA) National Bridge Inventory (NBI) fields used in BridgeRisk.

---

## 1. Bridge Identification
| Field Name | NBI Item | Description | Format / Units |
|:---|:---:|:---|:---|
| `STATE_CODE_001` | 1 | State FIPS code (23 = Maine, 15 = Hawaii, 10 = Delaware) | 2-digit string |
| `STRUCTURE_NUMBER_008` | 8 | Unique structure identifier within state | Text string |
| `bridge_id` | Derived | State code + Structure number compound key | E.g., `ME_0833` |

---

## 2. Condition Ratings (Target & Primary Predictors)
Evaluated on the standard FHWA 0–9 integer rating scale:
- **7–9:** Good condition (minor or no maintenance needed)
- **5–6:** Fair condition (sound structural elements with minor section loss)
- **0–4:** Poor condition (advanced deterioration, section loss, structural deficiency)
- **N:** Not applicable (converted to `NaN` during data preparation)

| Field Name | NBI Item | Component Evaluated |
|:---|:---:|:---|
| `DECK_COND_058` | 58 | Overall bridge deck condition rating |
| `SUPERSTRUCTURE_COND_059` | 59 | Structural support elements (girders, trusses, beams) |
| `SUBSTRUCTURE_COND_060` | 60 | Piers, abutments, bents, and footings |
| `CULVERT_COND_062` | 62 | Culvert barrel and headwall condition (if applicable) |

**Target Variable:**
- `target_deck_poor_next_year`: Binary indicator set to **1** if `DECK_COND_058` in year $t+1$ is $\le 4$ (Poor), and **0** if $> 4$ (Fair or Good).

---

## 3. Geometric & Structural Attributes
| Field Name | NBI Item | Description | Units |
|:---|:---:|:---|:---|
| `YEAR_BUILT_027` | 27 | Year structure was constructed | Calendar year |
| `bridge_age` | Derived | Inspection year minus `YEAR_BUILT_027` | Years |
| `MAIN_UNIT_SPANS_045` | 45 | Number of spans in main unit | Integer count |
| `MAX_SPAN_LEN_MT_048` | 48 | Length of maximum span | Meters |
| `STRUCTURE_LEN_MT_049` | 49 | Total structure length | Meters |
| `ROADWAY_WIDTH_MT_051` | 51 | Bridge roadway width, curb-to-curb | Meters |
| `DECK_WIDTH_MT_052` | 52 | Out-to-out deck width | Meters |
| `DECK_AREA` | Derived | Deck width $\times$ Structure length | Square meters |
| `TRAFFIC_LANES_ON_028A` | 28A | Number of traffic lanes on bridge | Integer count |

---

## 4. Traffic & Operational Loading
| Field Name | NBI Item | Description | Units |
|:---|:---:|:---|:---|
| `ADT_029` | 29 | Average Daily Traffic | Vehicles per day |
| `PERCENT_ADT_TRUCK_109` | 109 | Percentage of ADT representing truck traffic | Percent (0–99%) |
| `OPERATING_RATING_064` | 64 | Operating load rating (maximum permissible load) | Metric tons / rating factor |
| `INVENTORY_RATING_066` | 66 | Inventory load rating (customary live load capacity) | Metric tons / rating factor |

---

## 5. Material, Design & Foundation Risk
| Field Name | NBI Item | Description / Key Categories |
|:---|:---:|:---|
| `STRUCTURE_KIND_043A` | 43A | Material type (1=Concrete, 2=Concrete continuous, 3=Steel, 4=Steel continuous, 5=Prestressed concrete, 6=Prestressed continuous, 7=Timber) |
| `STRUCTURE_TYPE_043B` | 43B | Design type (1=Slab, 2=Stringer/Multi-beam, 3=Girder/Floorbeam, 4=Tee beam, 5=Box girder) |
| `SCOUR_CRITICAL_113` | 113 | Scour evaluation rating (1, 2, T = Scour critical; 8 = Stable; N = Not over water) |
| `WATERWAY_EVAL_071` | 71 | Waterway adequacy evaluation |
| `DECK_PROTECTION_108C` | 108C | Type of deck protection system (epoxy coating, cathodic protection, membrane) |
