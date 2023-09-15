# Define an empty list to store the file contents


def coco_names():
    names = []

    # Specify the path to your text file
    file_path = "coco.names.txt"

    # Open and read the file line by line
    with open(file_path, "r") as file:
        for line in file:
            # Append each line (as a string) to the list
            names.append(line.strip())  # Use .strip() to remove leading/trailing whitespace


    return names
