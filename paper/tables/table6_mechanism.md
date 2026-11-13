# Table 6: External-degradation mechanism summary (eps 0.0)

- ITA proxy overlap HAM↔Fitz: 0.727
- Prevalence: HAM train 0.111 → Fitz 0.528
- Fitz share per HAM ITA bin: bin0=0.17, bin1=0.25, bin2=0.34, bin3=0.21, bin4=0.03
- FST 5-6 share per HAM bin: bin0=0.39, bin1=0.33, bin2=0.19, bin3=0.06, bin4=0.03

| Method | pred-pos (dark) | prev (dark) | pred-pos (light) | prev (light) | sens dark | spec dark |
|---|---|---|---|---|---|---|
| class_strat | 0.464 | 0.510 | 0.326 | 0.531 | 0.492 | 0.565 |
| erm | 0.571 | 0.510 | 0.425 | 0.531 | 0.599 | 0.458 |
| proxy_dro | 0.452 | 0.510 | 0.311 | 0.531 | 0.462 | 0.559 |
| proxy_iw | 0.402 | 0.510 | 0.324 | 0.531 | 0.428 | 0.626 |
| proxy_strat | 0.415 | 0.510 | 0.307 | 0.531 | 0.442 | 0.613 |
| proxy_strat_joint | 0.624 | 0.510 | 0.470 | 0.531 | 0.640 | 0.394 |
| random_proxy_strat | 0.541 | 0.510 | 0.406 | 0.531 | 0.572 | 0.490 |