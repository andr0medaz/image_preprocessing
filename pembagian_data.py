from pathlib import Path
from ultralytics.data.split import autosplit

DATASET_DIR = Path("dataset-anotated-19-class")
RATIO_SPLIT = (0.8, 0.1, 0.1)
CLASS_NAMES = [
    "brotowali", 
    "bunga_telang", 
    "cabai_jawa",
    "cengkeh",
    "jahe_merah",
    "jambu_biji",
    "jeruk_purut",
    "kelor",
    "kencur",
    "kumis_kucing",
    "kunyit",
    "lengkuas",
    "lidah_buaya",
    "mengkudu",
    "pegagan",
    "sambiloto",
    "sereh",
    "sirih",
    "temulawak"
]

images_dir = DATASET_DIR / "images"
labels_dir = DATASET_DIR / "labels"

autosplit(path=str(images_dir), weights=RATIO_SPLIT, annotated_only=False)

yaml_lines = [
    f"path: {DATASET_DIR.resolve().as_posix()}",
    "train: autosplit_train.txt",
    "val: autosplit_val.txt",
    "test: autosplit_test.txt",
    "names:",
]

yaml_lines += [f"{i}: {n}" for i, n in enumerate(CLASS_NAMES)]
(DATASET_DIR / "data.yaml").write_text("\n".join(yaml_lines) + "\n")
print(f"\ndata.yaml dibuat: {DATASET_DIR / 'data.yaml'}")