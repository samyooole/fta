from msilib.schema import Feature
import pandas as pd
from sklearn.cluster import ward_tree, FeatureAgglomeration
import sys
sys.path.append("scripts/model/")
from cleanandTransform import cleanandTransform

df = pd.read_csv("core_excels/totaPlus_MandatoryPredictHF.csv")

text = df.text.astype('str')

text = text[0:50]

cat = cleanandTransform(filters = [], transformer_package='all-MiniLM-L12-v2')
cat.set_filters(['to_lower', 'remove_url','remove_special_character', 'normalize_unicode', 'remove_whitespace'])

cat.init_text(text)
cat.process_text()
cat.transform_text()
embed = cat.current_embedding


model = FeatureAgglomeration()






import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)



X = cat.current_embedding

# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(n_clusters=2, affinity='euclidean')

labels = model.fit_predict(X)
model = model.fit(X)
plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode="level", p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()