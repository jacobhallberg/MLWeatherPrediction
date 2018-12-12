import albumentations as Augment
from tensorflow.keras.utils import Sequence
import numpy as np

# Transformation function.
def augment_images():
    return Augment.Compose([
        Augment.GaussNoise(),
        Augment.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.02, rotate_limit=5),
        Augment.RandomBrightness()
    ])

# Generator that applies transformations to data and feeds the data to the neural network.
class WeatherImageGenerator(Sequence):
    def __init__(self, images, labels, batch_size=32, shuffle=True):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.images) / int(self.batch_size)))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]

        augmentation = augment_images()
        images = np.array([augmentation(image=self.images[i])["image"] for i in indices])
        labels = self.labels[indices]

        return images, labels
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.images))

        if self.shuffle: np.random.shuffle(self.indices)