import os


DATASET_PATH = "Dataset"


print("Class | Image Count")
print("-" * 30)

total = 0
for cls in sorted(os.listdir(DATASET_PATH)):
    cls_path = os.path.join(DATASET_PATH, cls)
    if os.path.isdir(cls_path):
        count = len(os.listdir(cls_path))
        total += count
        status = "LOW" if count < 100 else "OK"
        print(f"{cls:<20} {count:>5} images  {status}")

print("-" * 30)
print(f"{'TOTAL':<20} {total:>5} images")
