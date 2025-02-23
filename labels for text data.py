import pandas as pd

text_data = pd.read_csv(r"C:\Users\sunwa\OneDrive\Desktop\Neke Hackathon\training_data.csv")

text_data["text_id"] = range(1, len(text_data) + 1)

text_data.to_csv("text_training_data_with_ids.csv", index=False)