# Image embeddings laboratory

In this laboratory we aim to convert all images from MIT-67 dataset to an embedding representation using a pre-trained deep model. We are going to use the deep architecture vgg16 pre-trained using the ImageNet dataset. Then, we are going to visualize the embedding space generated and to apply clustering over the embedding representation.

Image embeddings laboratory includes 3 python scripts:

 - image_embeddings.py: To convert MIT-67 images to embeddings using vgg16 architecture pre-trained using ImageNet.
 - visualization.py: To visualize the embedding space generated.
 - clustering.py: To apply a clustering technique over the embeddings.

These scripts are on the ```https://github.com/UPC-MAI-DL/UPC-MAI-DL.github.io.git/_codes/3.EMB/Image_Emb``` folder, in the official Github repository. Download the folder and save a copy into your $HOME folder at GPFS system.

Before running the first script (i.e., image_embedding_lab.py), we have to download vgg16-ImageNet weights file and the MIT-67 training dataset. To download the vgg16 weights file and copy it into GPFS:

```
wget https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5
scp vgg16_weights_tf_dim_ordering_tf_kernels.h5 USERNAME@dt01.bsc.es:.keras/models/.
```

On the other hand, we have to also download MIT-67 datatset and copy it into GPFS:

```
wget http://147.83.200.110:8000/static/ferran/mit67_img_train.tar.gz
scp mit67_img_train.tar.gz USERNAME@dt01.bsc.es:.
```

And decompress it inside the GPFS system:

```
ssh USERNAME@mt1.bsc.es
tar -xzvf mit67_img_train.tar.gz
```

So now, we are ready to execute the first script. In order to do that, we have to request the execution through a job. Job requesting is done with launcher files. For example, to ask for a job to execute the ```image_embeddings.py```, we have to execute the following code:

```
mnsubmit image_embeddings.cmd
```

Then, we can keep control of our job running command:

```
bjobs
```

Once our job has execute our ```image_embeddings.py``` script, it will output a numpy file containing all image embeddings called ```mit67_embeddings.npz```. This file is going to be used as input by following scripts.

Visualization and clustering scripts and launchers follow the same procedure.
Check all python codes carefully to fully understand what they are doing.