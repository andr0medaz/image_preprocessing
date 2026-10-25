import cv2
import numpy as np
import os
import json

def letterbox_resize(img, target_size=640, pad_color=(114,114,114)):
    """
    - Resize gambar ke target_size x target_size dengan letterbox.
    - Pada data primer dimana gambar memiliki aspect ratio 1:1, padding otomatis = 0.
    - Sedangkan pada data sekunder dimana gambar memiliki aspect ratio yang beragam, 
      padding ditambahkan agar gambar hasil pemerataan tetap sesuai bentuk aslinya
    """
    height, width = img.shape[:2]
    scale = target_size / max(height, width)
    new_width, new_height = int(round(width * scale)), int(round(height * scale))

    # Interpolasi berdasarkan arah resize
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    img_resized = cv2.resize(img, (new_width, new_height), interpolation=interp)

    # Hitung padding
    padding_width = target_size - new_width
    padding_height = target_size - new_height
    top, bottom = padding_height // 2, padding_height - padding_height // 2
    left, right = padding_width // 2, padding_width - padding_width // 2

    img_padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=pad_color
    )

    metadata = {
        "original_size": [width, height],
        "scale": scale,
        "pad_left": left,
        "pad_right": right,
        "pad_top": top,
        "pad_bottom": bottom,
        "new_size_before_pad": [new_width, new_height]
    }

    return img_padded, metadata

def process_dataset(input_dir, output_dir, target_size=640, source_label="unknown"):
    os.makedirs(output_dir, exist_ok=True)
    metadata_all = []

    valid_ext = ('.jpg', '.jpeg', '.png', '.webp')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_ext)]

    for fname in files:
        img_path = os.path.join(input_dir, fname)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Gagal membaca: {fname}, dilewati")
            continue

        img_processed, meta = letterbox_resize(img, target_size)

        # Simpan gambar hasil
        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, img_processed)

        meta["filename"] = fname
        meta["source"] = source_label
        metadata_all.append(meta)

    return metadata_all

# Menjalankan kedua fungsi letterbox_resize dan process_dataset
meta_primer = process_dataset(
    "dataset_primer/", "data_processed/",
    target_size=640, source_label="primer"
)

meta_sekunder = process_dataset(
    "dataset_sekunder/", "data_processed",
    target_size=640, source_label="sekunder"
)

# Menyimpan metadata gabungan
all_metadata = meta_primer + meta_sekunder
with open("data_processed/resize_metadata.json", "w") as f:
    json.dump(all_metadata, f, indent=2)

print(f"Total diproses: {len(all_metadata)} gambar")

def validate_letterbox_output(output_dir, target_size=640):
    """Memastikan semua gambar hasil sudah tepat 640x640"""
    files = [f for f in os.listdir(output_dir) if f.lower().endswith(('.jpg', '.png'))]
    errors = []
    
    for fname in files:
        img = cv2.imread(os.path.join(output_dir, fname))
        if img.shape[0] != target_size or img.shape[1] != target_size:
            errors.append(fname)
    
    if errors:
        print(f"Ditemukan {len(errors)} gambar dengan ukuran tidak sesuai: {errors}")
    else:
        print(f"Semua {len(files)} gambar sudah valid {target_size}x{target_size}")

validate_letterbox_output("data_processed/", 640)