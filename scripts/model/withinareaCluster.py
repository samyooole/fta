from sklearn.cluster import DBSCAN


# try DBSCAN

clusterModel = DBSCAN(eps=0.15).fit(para_array)
labels = pd.Series(clusterModel.labels_)
garble = pd.Series(sentences)
legible = pd.Series([understander[tuple(sentence)] for sentence in sentences])

readable = pd.concat([labels, garble, legible], axis=1)
readable.columns = ['labels', 'garble', 'legible']

readable[readable['labels'] == -1]