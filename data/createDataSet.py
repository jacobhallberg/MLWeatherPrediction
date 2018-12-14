from glob import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

image_size = (500,320)
image_types = ["DAA", "N0S", "DTA", "N0Z", "N1U"]
base_directory = "DAA"

# Assumes image directories are in the current working directory.
filenames = sorted(glob(base_directory  + "/*"))

# imgs = np.load("images.npy")
# print(imgs[1][1].dtype)
# print(imgs.shape)

# plt.imshow(imgs[140][0])
# print(imgs[140][0])
# plt.show()

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
count = 0
for image_filename in tqdm(filenames):
    if image_filename[8:-6] not in weather_descriptions: 
        count += 1
        continue
    else: matching_weather_descriptions.append(weather_descriptions[image_filename[8:-6]])

    collective_images_at_one_time = []
    for image_type in image_types:
        try:
            image = Image.open(image_filename.replace(base_directory, image_type))
            image = image.convert("L") # Converts to black and white.
            image = image.resize(image_size, Image.ANTIALIAS)
            collective_images_at_one_time.append(np.asarray(image))
        except:
            # Corresponding filename was not, create empty black image instead of loading an image.
            image = Image.new("L", image_size) # Create empty black image.
            collective_images_at_one_time.append(np.asarray(image))

    images.append(np.array(collective_images_at_one_time))

images = np.array(images)
np.save("images.npy", images)

matching_weather_descriptions = np.array(matching_weather_descriptions)
np.save("labels.npy", matching_weather_descriptions)

print(count)
