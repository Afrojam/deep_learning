import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Load dataset embeddings and labels.
obj = np.load('mit67_embeddings.npz')
labels = obj['labels']
embeddings = obj['embeddings']

# Set up the subset of labels to visualize.
n_labels = 5
_uniq = np.unique(labels)[:n_labels]
print('The {n} labels selected to visualize:'.format(n=n_labels), _uniq)

# Create the embeddings matrix (and its corresponding labels list) 
# containing instances belonging to labels in the previous subset.
subsets_emb = np.zeros((0,4096))
lab_range = {}
offset = 0
for idx, _lab in enumerate(_uniq):
    part_emb = embeddings[np.where(labels == _lab)]
    subsets_emb = np.concatenate((subsets_emb, part_emb))
    lab_range[_lab] = [offset, offset+part_emb.shape[0]]
    offset += part_emb.shape[0]
print('subsets_emb shape:', subsets_emb.shape)

# Apply a dimensionality reduction technique to visualize 2 dimensions.
vis_matrix = TSNE(n_components=2).fit_transform(subsets_emb)

# Using matplotlib to create the plot and show it!
cmap = plt.get_cmap('gist_rainbow')
colors = [cmap(i) for i in np.linspace(0, 1, n_labels)]
plt.figure()
for idx, _lab in enumerate(_uniq):
    r = lab_range[_lab]
    plt.scatter(vis_matrix[r[0]:r[1],0], vis_matrix[r[0]:r[1],1], s=30, c=colors[idx], label=_lab)
plt.legend(loc='best')
plt.savefig('vis_img.pdf')
plt.show()
