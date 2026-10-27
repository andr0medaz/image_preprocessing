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

labels_df.to_csv("labels_df.csv")