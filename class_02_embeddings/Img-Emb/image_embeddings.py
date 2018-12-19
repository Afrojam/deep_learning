import sys, time
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from glob import glob

# Define a dataset iterator by batches of an specific size.
def input_pipeline(image_files, batch_size):
    end_flag = False
    for n in range(len(image_files)):
        x_batch = np.zeros((0,224,224,3))
        y_batch = []
        for i in range(batch_size):
            try:
                img_path = image_files.pop(0)
            except IndexError:
                end_flag = True
                break
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x_batch = np.concatenate((x_batch, x), axis=0)
            y = img_path.split('/')[-2]
            y_batch.append(y)
        if end_flag:
            if x_batch.shape[0]>0:
                yield x_batch, y_batch
        else:
            yield x_batch, y_batch


# Load the vgg16 architecture with pre-trained weights using ImageNet2012 dataset.
base_model = VGG16(weights='imagenet')
print('Layers of vgg16 architecture:', [x.name for x in base_model.layers])
print('We are building the embedding using activations from layer fc2 (i.e., second fully-connected).')
sys.stdout.flush()
# Define a custom model that given an input, outputs activations from requested layer.
model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)

# Defining variables where to iteratively save dataset embeddings and labels.
dataset_emb = np.zeros((0,4096))
dataset_lab = []

print('Processing image embeddings through batches of 10 images per step.')
sys.stdout.flush()
# Create a list containing all image_paths. 
image_files = glob('~/mit67_img_train/*/*')
step = 0
tot = len(image_files)/10
if len(image_files)%10 > 0:
    tot += 1
# Batching loop.
for x_batch, y_batch in input_pipeline(image_files, 10):
    t0 = time.time()
    # Preprocessing input images for the vgg16 model.
    x = preprocess_input(x_batch)
    # Obtain the embeddings of current batch of images.
    batch_emb = model.predict(x)
    dataset_emb = np.concatenate((dataset_emb, batch_emb))
    dataset_lab += y_batch

    step += 1
    print('step: {s}/{tot} in {t}s'.format(s=step, tot=tot, t=time.time()-t0))
    sys.stdout.flush()

print('Dataset embedding shape:', dataset_emb.shape)
print('Length of dataset labels', len(dataset_lab))
print('Saving dataset embeddings into mit67_embeddings.npz file...')
sys.stdout.flush()
np.savez('mit67_embeddings.npz', embeddings=dataset_emb, labels=dataset_lab)

