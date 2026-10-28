from collections import Counter
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import KFold
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.split import autosplit
from ultralytics.data.utils import check_det_dataset

ROOT = Path("dataset-anotated-percobaan-2").resolve()
ksplit = 5

"""
Split terlebih dahulu, pada dokumentasi kfold ultralytics dataset yang digunakan sudah dalam kondisi terbagi
antara data train, val, dan test (https://docs.ultralytics.com/guides/kfold-cross-validation)
"""
autosplit(path=ROOT / "images", weights=(0.7, 0.1, 0.2), annotated_only=True)

"""
Membuat data.yaml berdasarkan classes.txt hasil export dari label studio
"""
names = [l.strip() for l in (ROOT / "classes.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
yaml_file = ROOT / "data.yaml"
yaml_file.write_text(
    yaml.safe_dump(
        {
            "path": ROOT.as_posix(),
            "train": "autosplit_train.txt",
            "val": "autosplit_val.txt",
            "test": "autosplit_test.txt",
            "names": dict(enumerate(names)),
        },
        allow_unicode=True,
        sort_keys=False,
    ),
    encoding="utf-8",
)

"""
Load dataset (train dan validation) tanpa menyentuh data test
"""
data = check_det_dataset(str(yaml_file))
dataset_path = Path(data["path"])

sources = []
for split in ("train", "val"):
    source = data.get(split)
    if source:
        sources.extend(source if isinstance(source, list) else [source]) 

dataset = YOLODataset(sources, data=data, augment=False)
images = [Path(p) for p in dataset.im_files]

classes = data["names"]
cls_idx = sorted(classes.keys())

labels_df = pd.DataFrame(0.0, columns=cls_idx, index=images)
for image, label in zip(images, dataset.labels):
    label_counter = Counter(label["cls"].flatten().astype(int))
    labels_df.loc[image, list(label_counter)] = list(label_counter.values())

"""
K-Fold dataset split
"""
# situasional labels karena tidak run dalam sekali jalan
# path_labels = ROOT / "labels_df.csv"
# labels_df = pd.read_csv(path_labels, index_col=0)

ksplit = 5
kf = KFold(n_splits=ksplit, shuffle=True, random_state=20)
kfolds = list(kf.split(labels_df))

folds = [f"split_{n}" for n in range(1, ksplit + 1)]
folds_df = pd.DataFrame(index=labels_df.index, columns=folds)

for i, (train, val) in enumerate(kfolds, start=1):
    folds_df.loc[labels_df.index[train], f"split_{i}"] = "train"
    folds_df.loc[labels_df.index[val], f"split_{i}"] = "val"

# check distribusi class label pada setiap fold sebagai ratio antara class yang ada di val dengan class yang di train
fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)

for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
    train_totals = labels_df.iloc[train_indices].sum()
    val_totals = labels_df.iloc[val_indices].sum()

    # To avoid division by zero, we add a small value (1E-7) to the denominator
    ratio = val_totals / (train_totals + 1e-7)
    fold_lbl_distrb.loc[f"split_{n}"] = ratio

# save path txt dan yaml per fold
save_path = dataset_path.parent / f"{ksplit}-Fold_Cross-val"
save_path.mkdir(parents=True, exist_ok=True)
ds_yamls = []

for split in folds_df.columns:
    for partition in ("train", "val"):
        split_images = folds_df.index[folds_df[split] == partition]
        paths = "\n".join(map(str, split_images))
        (save_path / f"{split}_{partition}.txt").write_text(f"{paths}\n")

    dataset_yaml = save_path / f"{split}.yaml"
    ds_yamls.append(dataset_yaml)
    with open(dataset_yaml, "w") as ds_y:
        yaml.safe_dump(
            {
                "path": save_path.as_posix(),
                "train": f"{split}_train.txt",
                "val": f"{split}_val.txt",
                "names": classes,
            },
            ds_y,
        )

# opsional untuk di save
labels_df.to_csv("labels_df.csv")
fold_lbl_distrb.to_csv(save_path / "kfold_label_distribution.csv")
# wajib di save
folds_df.to_csv(save_path / "kfold_datasplit.csv")