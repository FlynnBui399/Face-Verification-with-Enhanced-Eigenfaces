# Dataset

Dataset files are **not** included in this repo (they are too large). You need to download them and place them in the following structure:

```
dataset/
├── casia-webface/          # Training
│   ├── train.rec
│   ├── train.idx
│   └── property
├── eval/                    # Verification benchmarks
│   ├── lfw.bin
│   ├── cfp_fp.bin
│   └── agedb_30.bin
└── cache/                   # Created at runtime (preprocessing cache)
```

- **CASIA-WebFace:** RecordIO format (train.rec, train.idx, property).  
- **LFW / CFP-FP / AgeDB-30:** InsightFace `.bin` format (image list + `issame_list`).

See the root **README.md** for full setup and usage instructions.
