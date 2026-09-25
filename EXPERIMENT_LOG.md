# Experiment Log

This log tracks every training and validation run, hyperparameter configuration, metric outcome, and changes made across the project.

| Run ID | Date | Task | Model / Config | Key Hyperparameters | Metric Results (Train/Val/Test) | Notes & Observations |
|---|---|---|---|---|---|---|
| M1-DATA-01 | 2026-09-25 | Data Pipeline Validation | Oxford-IIIT Pet Manifest Generator | Seed: 42, Split: 80% train / 20% val. Corruptions: 4 classes (clean, s&p, blur, occlusion) | OxfordPet Split: Official trainval split into 80% train / 20% val; Test set isolated. | Deterministic manifests saved to `configs/manifests/`. Reusable across Tasks 1-3. |
| M1-DATA-02 | 2026-09-25 | Data Pipeline Validation | FS2K Manifest Generator | Seed: 42, Base: official train/test (anno_train.json / anno_test.json), 15% stratified val by style (0/1/2) | FS2K Split: 1,058 train pairs (899 train / 159 val stratified) + 1,046 test pairs = 2,104 pairs total. | Style distribution preserved across train and val splits. |
| M1-CORRUPT-01 | 2026-09-25 | Corruption Verification | Runtime & Deterministic Generators | S&P: p in [0.02, 0.15]; Blur: k in {3,5,7}, sigma in [0.5, 2.5]; Occlusion: 1-3 masks (10-35% area). Fixed Test: S&P (0.03, 0.08, 0.15), Blur ((3,0.7), (5,1.5), (7,2.5)), Occlusion (1/2/3 masks ~10/20/35%). | All 4 classes produced with equal probability at runtime. Deterministic test manifests verified. | Verified corruptions visual inspection and array bounds [0, 1]. |
