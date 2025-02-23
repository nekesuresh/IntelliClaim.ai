import os
import pandas as pd

image_folder = r"C:\Users\sunwa\OneDrive\Desktop\Neke Hackathon\train\images"

image_files = os.listdir(image_folder)

image_data = pd.DataFrame({
    "image_id": range(1, len(image_files) + 1),
    "image_path": [os.path.join(image_folder, f) for f in image_files]
})

image_data.to_csv("image_data_with_ids.csv", index=False)