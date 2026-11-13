# Table 1: Dataset integrity audit

| Check | HAM10000 | Fitzpatrick17k (eval subset) |
|---|---|---|
| Images | 10,015 (7,470 lesions) | 3,190 |
| Exact content duplicates | 2 groups (4 images) | 0 |
| Duplicates spanning splits | 0 | n/a (external set) |
| Lesion label inconsistencies | 0 | 0 (per-key) |
| Content md5 == filename | n/a | 3190/3190 |
| Positive prevalence | 0.111 (melanoma) | 0.528 (malignant) |
| Skin-tone annotation | none (by design) | FST 1-6 + unknown |
| Cross-dataset content overlap | 0 | 0 |
| QC-flagged rows | n/a | 86 / 3190 |

Subgroup counts (eval subset): FST 0 = 104, FST 1 = 696, FST 2 = 1062, FST 3 = 641, FST 4 = 432, FST 5 = 187, FST 6 = 68.

Audit correction: `fst == 0` in the re-host corresponds to the original `fitzpatrick_scale == -1` (unknown); the light group is FST 1-2 only (n = 1758); FST 0 excluded from binary contrasts.