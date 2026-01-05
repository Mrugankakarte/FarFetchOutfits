# FarFetchOutfits
My solution to Farfetch Outfits Challenge (https://eval.ai/web/challenges/challenge-page/1721/overview) using Graph Neural Networks

Challenge Readme: [README](Challenge README.md)

```
@inproceedings{FarfetchSIGIR2022,
author = {Baia, Luis and Martins, Eder and Goncalves, Diogo and Marinho, Vanessa and Viegas, Felipe and Silva, Ana and Otto, Tiago},
title = {Farfetch Outfits Challenge on E-Commerce Workshop},
year = {2022},
booktitle = {SIGIR eCom 2022}
}
```


## Validation Metrics

**BPR loss**

out=hid-in_channesls = 64

heads=8

candidates = 12

### **candidates from same category**:

| k  | MAP@k     | HitRate@k |
|----|-----------|-----------|
| 1  | 0.2371    | 0.2371    |
| 2  | 0.31905   | 0.401     |
| 3  | 0.358583  | 0.5196    |
| 5  | 0.396423  | 0.686     |
| 10 | 0.42877   | 0.9263    |

### **candidates not from same category**:

| k  | MAP@k    | HitRate@k |
|----|----------|-----------|
| 1  | 0.4762   | 0.4762    |
| 2  | 0.54495  | 0.6137    |
| 3  | 0.567683 | 0.6819    |
| 5  | 0.591668 | 0.7879    |
| 10 | 0.616826 | 0.9746    |


## Sample Outputs

![](Outputs\SimpleGAT\samecat_1.png)
![](Outputs\SimpleGAT\samecat_2.png)
![](Outputs\SimpleGAT\samecat_3.png)

![](Outputs\SimpleGAT\notsamecat_1.png)
![](Outputs\SimpleGAT\notsamecat_2.png)
![](Outputs\SimpleGAT\notsamecat_3.png)
