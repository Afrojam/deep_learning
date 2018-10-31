import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.decomposition import PCA

# Load dataset embeddings and labels.
obj = np.load('mit67_embeddings.npz')
labels = obj['labels']
embeddings = obj['embeddings']

# Set up the subset of labels to visualize.
n_labels = 5
_uniq = np.unique(labels)[:n_labels]
print('The {n} labels selected to clusterize:'.format(n=n_labels), _uniq)

# Create the embeddings matrix (and its corresponding labels list)
# containing instances belonging to labels in the previous subset.
subsets_emb = np.zeros((0,4096))
true_labels = []
for idx, _lab in enumerate(_uniq):
    part_emb = embeddings[np.where(labels == _lab)]
    subsets_emb = np.concatenate((subsets_emb, part_emb))
    true_labels += [_lab for _ in range(part_emb.shape[0])]
true_labels = np.array(true_labels)

# Use a dimensionality reduction technique (e.g., PCA).
reduced_emb = PCA(n_components=100).fit_transform(subsets_emb)

# Apply a clustering technique (e.g., KMeans) and evaluate its performance with
# a metric (e.g., Normalized Mutual Information).
pred_labels = KMeans(n_clusters=n_labels, random_state=0).fit_predict(reduced_emb)
nmi = NMI(true_labels, pred_labels)
print('NMI score:', nmi)
