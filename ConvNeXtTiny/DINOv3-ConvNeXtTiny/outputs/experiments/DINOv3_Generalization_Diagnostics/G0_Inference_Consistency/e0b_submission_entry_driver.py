"""Run the packaged submission inference entry (E0-B) on a Study manifest."""
import csv
import json
import sys
import time
from pathlib import Path

manifest_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
package_dir = Path(sys.argv[3])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
sys.path.insert(0, str(package_dir))
from dino_feature_inference import FeaturePredictor

predictor = FeaturePredictor(package_dir)
rows = []
started = time.perf_counter()
for record in manifest["studies"]:
    study_started = time.perf_counter()
    probabilities = predictor.predict_study_paths(record["image_paths"])
    rows.append({
        "Index": int(record["index"]),
        "Subject_id": record["subject_id"],
        "Study_id": record["study_id"],
        "Image_Count": int(record["image_count"]),
        **{f"prob_{index}": float(probabilities[index]) for index in range(10)},
        "Inference_Ms": (time.perf_counter() - study_started) * 1000.0,
    })
elapsed = time.perf_counter() - started
with output_path.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({
    "studies": len(rows),
    "elapsed_seconds": elapsed,
    "mean_study_ms": elapsed / max(len(rows), 1) * 1000.0,
}))
