# Trained checkpoints

[Download](https://drive.google.com/drive/folders/1YFTKxH1YVJHKMlWIu16oKxbJewJGXuvi?usp=sharing) the folder `checkpoints/` and place it in the repository root. Each
`best_model.pt` holds the config, the EMA weights (fp16), the epoch and the training options.

| Checkpoint | Split | Model in the paper | Options | Epoch |
|---|---|---|---|---|
| `a2holo_seed1` | D3PM | A2Holo, released checkpoint | the recipe, `--seed 1` | 640 |
| `splitA_recipe` | protein-disjoint | A2Holo | the recipe | 95 |
| `splitA_depth4` | protein-disjoint | Ankh3, 4+4 | `--depth_double 4 --depth_single 4` | 55 |
| `splitA_no_plm` | protein-disjoint | none, 0+12 | `--no_plm --depth_double 0` | 170 |
| `splitA_seq_from_structure` | protein-disjoint | structure copy | `--seq_from_structure --no_global_token` | 110 |
| `splitA_cross_attention` | protein-disjoint | Ankh3, cross | `--attention cross` | 90 |
| `splitA_esm2` | protein-disjoint | ESM-2 | `--plm esm2` | 75 |
| `splitA_esmc` | protein-disjoint | ESM-C | `--plm esmc` | 205 |

```bash
python scripts/evaluate.py --checkpoint checkpoints/a2holo_seed1/best_model.pt
python scripts/evaluate.py --checkpoint checkpoints/splitA_recipe/best_model.pt --splits data/apollodiff_splitA.json
```

`splitA_esm2` and `splitA_esmc` need `ESM=1 examples/preprocess.sh`.

| File | SHA-256 |
|---|---|
| `a2holo_seed1/best_model.pt` | `2f19784feea994ed4ffb9cdce1eaa0e3b5f5965fcae12ed6c64317e8c64a489b` |
| `splitA_recipe/best_model.pt` | `37ed1c55f7ed48c43fd45cf5587eeaa5addbfd012e02e250c8abf798b7a70aaf` |
| `splitA_depth4/best_model.pt` | `49bc3a608bebdffa2b595b56c6aeb7748e3c367a6a73c88c4ed7e12fce8709dd` |
| `splitA_no_plm/best_model.pt` | `a0b5fa7d8b92b0d02d3df2310d6e106d0f3ec684233e01a7d9818b99135ec968` |
| `splitA_seq_from_structure/best_model.pt` | `7c4c704acd2c98e8e1cdb64d04ae4941151b27d377702bbf413f37d06706215c` |
| `splitA_cross_attention/best_model.pt` | `937913a1fc10376ed7514736b1e277c613841facc2104a9021fafa32b502731b` |
| `splitA_esm2/best_model.pt` | `0eb199862222905873dd3b1950fdd70d230ae8435b2d4c4cc45327420b2fc68e` |
| `splitA_esmc/best_model.pt` | `08c2e4157beedd5be89ee9d9055392a5c2f567ba49e1edf4f530f3b798f64d81` |
