import os

# Set the directories for train, test, and valid folders
directories = [r"C:\Users\sunwa\OneDrive\Desktop\Neke Hackathon\train",
               r"C:\Users\sunwa\OneDrive\Desktop\Neke Hackathon\test",
               r"C:\Users\sunwa\OneDrive\Desktop\Neke Hackathon\valid"]

def update_class_ids(label_file_path):
    # Open and read the label file
    with open(label_file_path, 'r') as file:
        lines = file.readlines()

    # Create a list to store the updated lines
    updated_lines = []

    # Iterate through each line (label entry)
    for line in lines:
        # Split the line into components (class_id, x_center, y_center, width, height)
        components = line.strip().split()

        # Check if the class_id is '2' and update it to '0'
        if components[0] == '2':
            components[0] = '0'

        # Append the updated line to the list
        updated_lines.append(" ".join(components) + "\n")

    # Rewrite the label file with the updated class ids
    with open(label_file_path, 'w') as file:
        file.writelines(updated_lines)

# Loop through each directory and process the label files
for directory in directories:
    # Iterate over each label file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):  # YOLO label files have .txt extension
            label_file_path = os.path.join(directory, filename)
            update_class_ids(label_file_path)
            print(f"Updated class ids in {label_file_path}")
