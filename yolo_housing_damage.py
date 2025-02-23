from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')

    results = model.train(data=r"C:\Users\MMA\Downloads\Damages.v14i.yolov8\data.yaml",
                      epochs = 100,
                      patience = 10,
                      batch = -1,
                      device = 'cuda',
                      optimizer = 'auto',
                      val = True,
                      plots= True,
                      save = True,
                      workers = 4,
                      verbose = True)
    result_prediction = model.predict(source="test_images/", conf=0.25)  # Adjust confidence as needed

    damage_probabilities = []
    for r in result_prediction:
        for i, box in enumerate(r.boxes):  # Iterate over detected objects
            damage_probabilities.append({
                "image_id": r.path.split("/")[-1],  # Extract image filename
                "damage_probability": box.conf.item()  # Confidence score of the detection
            })

    # Save to CSV
    damage_prob_df = pd.DataFrame(damage_probabilities)
    damage_prob_df.to_csv("damage_probabilities.csv", index=False)
    print("Damage probabilities saved to 'damage_probabilities.csv'.")
    
if __name__=="__main__":
    main()
