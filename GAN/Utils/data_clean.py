from PIL import Image
import pandas as pd
import os

# Read the image files in a folder
# Read the labels from a txt file
# The image file names are numbers, find the corresponding labels
# For example, the label file stores the valence values of all the images
# However, only a few images are used in the training set
# Find the corresponding labels for the training set images

# Read valence values from a txt file
valence_values = pd.read_csv(
    r"C:\Users\cxoox\Desktop\AIST3110_Project\GAN\Data\EmotionLabel\valence_avg_10766_v2.txt",
    header=None,
)
arousal_values = pd.read_csv(
    r"C:\Users\cxoox\Desktop\AIST3110_Project\GAN\Data\EmotionLabel\arousal_avg_10766_v2.txt",
    header=None,
)

# Read image file names from the folder
images = os.listdir(r"C:\Users\cxoox\Desktop\AIST3110_Project\GAN\Data\All_photos")
# Extract only the numeric part of the image file names (assuming they are numbers)
images = [int(img.split(".")[0]) for img in images if img.split(".")[0].isdigit()]
# Sort the image file names
images.sort()


# Create a dictionary to store valence values for the training set images
new_valence_values = {image: valence_values.iloc[image - 1, 0] for image in images}
new_arousal_values = {image: arousal_values.iloc[image - 1, 0] for image in images}


# save the dictionary to a new txt file with only the second column
# Save the dictionary to a new txt file
with open(
    r"C:\Users\cxoox\Desktop\AIST3110_Project\GAN\Data\all_photos_valence.txt", "w"
) as f:
    for key, value in new_valence_values.items():
        f.write(f"{value}\n")
with open(
    r"C:\Users\cxoox\Desktop\AIST3110_Project\GAN\Data\all_photos_arousal.txt", "w"
) as f:
    for key, value in new_arousal_values.items():
        f.write(f"{value}\n")


# Combine the two txt files into one csv file with coloumns "valence" and "arousal"
valence_df = pd.read_csv(
    r"C:\Users\cxoox\Desktop\AIST3110_Project\GAN\Data\all_photos_valence.txt",
    header=None,
)
arousal_df = pd.read_csv(
    r"C:\Users\cxoox\Desktop\AIST3110_Project\GAN\Data\all_photos_arousal.txt",
    header=None,
)
# convert images to a pandas DataFrame
images = pd.DataFrame(images, columns=["image"])
combined_df = pd.concat([images, valence_df, arousal_df], axis=1)
combined_df.columns = ["image", "valence", "arousal"]
combined_df.to_csv(
    r"C:\Users\cxoox\Desktop\AIST3110_Project\GAN\Data\all_photos_valence_arousal.csv",
    index=False,
    header=False,
)
