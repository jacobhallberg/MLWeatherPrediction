from glob import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd

image_size = (500,320)
image_types = ["DAA", "N0S"]
base_directory = "DAA"

# Assumes image directories are in the current working directory.
filenames = sorted(glob(base_directory  + "/*"))

# Create weather description data.
weather_descriptions = {}
weather_data = pd.read_csv("denver_data.csv")
weather_data = weather_data[["datetime", "weather_description_Denver"]]

for _, (datetime,weather_description) in weather_data.iterrows():
    datetime = datetime.replace("-", "").replace(" ", "").replace(":","")[:-4]
    weather_descriptions[datetime] = weather_description

# Create image data and match data with description.
images = []
matching_weather_descriptions = []
for image_filename in tqdm(filenames):
    if image_filename[8:-6] not in weather_descriptions: continue
    else: matching_weather_descriptions.append(weather_descriptions[image_filename[8:-6]])

    collective_images_at_one_time = []
    for image_type in image_types:
        try:
            image = Image.open(image_filename.replace(base_directory, image_type))
            image = image.convert("1") # Converts to black and white.
            image = image.resize(image_size, Image.ANTIALIAS)
            collective_images_at_one_time.append(np.asarray(image))
        except:
            # Corresponding filename was not, create empty black image instead of loading an image.
            image = Image.new("1", image_size) # Create empty black image.
            collective_images_at_one_time.append(np.asarray(image))

    images.append(np.array(collective_images_at_one_time))

images = np.array(images)
np.save("images.npy", images)

matching_weather_descriptions = np.array(matching_weather_descriptions)
np.save("labels.npy", matching_weather_descriptions)