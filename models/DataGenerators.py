import albumentations as Augment
from keras.utils import Sequence
import numpy as np
from skimage.transform import resize


# Transformation function.
def augment_images():
    return Augment.Compose([
        Augment.GaussNoise(),
        Augment.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.02, rotate_limit=5),
        Augment.RandomBrightness()
    ])

# Generator that applies transformations to data and feeds the data to the neural network.
class Generator(Sequence):
    def __len__(self):
        return int(np.ceil(len(self.images) / int(self.batch_size)))
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.images))

        if self.shuffle: np.random.shuffle(self.indices)

class WeatherImageGenerator(Generator):
    def __init__(self, images, labels, batch_size=32, shuffle=True):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]

        augmentation = augment_images()
        images = np.array([augmentation(image=self.images[i])["image"] for i in indices])
        labels = self.labels[indices]

        return images, labels

class SequenceWeatherImageGenerator(Generator):
    def __init__(self, images,batch_size=32, shuffle=True, sequence_size=3, label_size=(307, 487), class_index=1):
        self.images = images
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sequence_size = sequence_size
        self.label_size = label_size
        self.class_index = class_index
        self.on_epoch_end()
    
    # Grab series of sequential temporal images and the label for the n+1 timestep.
    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        
        images = []
        labels = []
        
        for i in indices:
            while(i + self.sequence_size >= self.images.shape[0] - 1): i -= 1
            
            images.append(np.array([self.images[j] for j in range(i, i+self.sequence_size)]))
            labels.append(resize(self.images[i+self.sequence_size+1][:,:,self.class_index], self.label_size))
            
#             label = np.array([resize(image, self.label_size)
#                               for image in np.moveaxis(self.images[i+self.sequence_size+1][:,:,self.class_index], -1, 0)])
#             label = np.moveaxis(label, 0, -1)
            
#             labels.append(label)
            
        images = np.array(images)
        labels = np.expand_dims(np.array(labels), axis=-1)
        
        
        return images, labels