import cv2
import random
import numpy as np
import albumentations as A
from pathlib import Path
from tqdm import tqdm

SEED = 42
random.seed(SEED); np.random.seed(SEED)

ROOT = Path(__file__).parent / "dataset-anotated-percobaan-1"
TRAIN_TXT = ROOT / "autosplit_train.txt"
OUT_IMG = ROOT / "aug" / "images"
OUT_LBL = ROOT / "aug" / "labels"
OUT_TXT = ROOT / "autosplit_train_aug.txt"
# jumlah versi augmentasi per gambar
N_AUG = 2

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)

# pipeline augmentasi menggunakan library albumentations
train_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_range=(-0.3, 0.3),
            contrast_range=(0.0, 0.0),
            p=0.8,
        ),
        A.RandomGamma(gamma_range=(80, 120), p=0.3),
        A.RandomShadow(
            shadow_roi=(0, 0, 1, 1),       
            p=0.3,
        ),
    ], bbox_params=A.BboxParams(
        coord_format='yolo',
        label_fields=['class_labels'],
    ),
)

# helper 
def resolve(line: str) -> Path:
    """Path di autosplit berbentuk './images/abc.jpg' (relatif dengan ROOT)."""
    line = line.strip()
    return ROOT / line[2:] if line.startswith("./") else Path(line)

# def img2label(p: Path) -> Path:
#     s = str(p)
#     return Path("/labels/".join(s.rsplit("/images/", 1))).with_suffix(".txt")
def img2label(p: Path) -> Path:
    s = p.as_posix()
    return Path("/labels/".join(s.rsplit("/images/", 1))).with_suffix(".txt")

def read_yolo(lbl_path: Path):
    boxes, classes = [], []
    if lbl_path.exists():
        for ln in lbl_path.read_text().splitlines():
            parts = ln.split()
            if len(parts) != 5:
                continue
            c, x, y, w, h = int(parts[0]), *map(float, parts[1:])
            # clip supaya tidak error karena nilai sedikit di luar [0,1]
            x, y = np.clip([x, y], 0, 1)
            w, h = np.clip([w, h], 1e-6, 1)
            boxes.append([x, y, w, h]); classes.append(c)
    return boxes, classes

# proses augmentasi beserta bar progres dengan tqdm
img_paths = [resolve(l) for l in TRAIN_TXT.read_text().splitlines() if l.strip()]
new_entries = []

for img_path in tqdm(img_paths):
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes, classes = read_yolo(img2label(img_path))

    for i in range(N_AUG):
        out = train_transform(image=img, bboxes=boxes, class_labels=classes)
        # semua bbox hilang akan dilewati
        if boxes and len(out["bboxes"]) == 0:
            continue 

        name = f"{img_path.stem}_aug{i}"
        cv2.imwrite(
            str(OUT_IMG / f"{name}.jpg"),
            cv2.cvtColor(out["image"], cv2.COLOR_RGB2BGR)
        )

        with open(OUT_LBL / f"{name}.txt", "w") as f:
            for c, (x, y, w, h) in zip(out["class_labels"], out["bboxes"]):
                f.write(f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        new_entries.append(f"./aug/images/{name}.jpg")

OUT_TXT.write_text("\n".join(new_entries) + "\n")
print(f"Gambar augmentasi: {len(new_entries)}")
