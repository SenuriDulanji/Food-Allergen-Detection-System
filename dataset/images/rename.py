import os

folder_path = "dataset\images\potato_curry"
prefix = "img"
start = 1

for i, filename in enumerate(os.listdir(folder_path), start):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        ext = os.path.splitext(filename)[1]
        new_name = f"{prefix}{i:03d}{ext}"
        os.rename(
            os.path.join(folder_path, filename),
            os.path.join(folder_path, new_name)
        )
