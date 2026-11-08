from pathlib import Path
from ultralytics.data.split import autosplit
import yaml

DATASET_DIR = Path("dataset-anotated-percobaan-3").resolve()

"""
split dengan library autosplit ultralytics
"""
autosplit(path=DATASET_DIR / "images", weights=(0.85, 0.14, 0.01), annotated_only=True)

"""
membuat data.yaml berdasarkan classes.txt hasil export dari label studio
"""
names = [l.strip() for l in (DATASET_DIR / "classes.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
yaml_file = DATASET_DIR / "data.yaml"
yaml_file.write_text(
    yaml.safe_dump(
        {
            "path": DATASET_DIR.as_posix(),
            "train": "autosplit_train.txt",
            "val":"autosplit_val.txt",
            "names": dict(enumerate(names)),
        },
        allow_unicode=True,
        sort_keys=False,
    ),
    encoding="utf-8",
)