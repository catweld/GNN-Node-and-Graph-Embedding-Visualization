import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.io import savemat
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import umap
from ivis import Ivis
import time
import numpy as np

def plot_embeddings(embeddings,labels):
  print_statistics(embeddings,labels)
  
  num_classes = len(set(labels))
  fig = plt.figure(figsize=(12,8), dpi=80)
  
  for class_id in range(num_classes):
    label_to_color_map = {0: "red", 1: "blue", 2: "green", 3: "orange", 4: "yellow", 5: "pink", 6: "gray", 7: "purple", 8: "cyan", 9: "olive", 10: "red"}
    plt.scatter(embeddings[labels == class_id, 0], embeddings[labels == class_id, 1], s=20, color=label_to_color_map[class_id], edgecolors='black', linewidths=0.2)
  plt.rcParams.update({'font.size': 28})
  plt.legend(label_to_color_map,fontsize='x-small')
  plt.show()

def reduce_dims(hidden_embeddings,labels,method,perplexity,neighbors,min_dist,supervised):

  if method == 'pca':
    start = time.time()
    embeddings = PCA(n_components=2).fit_transform(hidden_embeddings)
    end = time.time()
    print(method,'time',end - start)

  elif method == 'tsne':
    start = time.time()
    embeddings = TSNE(n_components=2, perplexity=perplexity, method='barnes_hut').fit_transform(hidden_embeddings)
    end = time.time()
    print(method,'time',end - start)

  elif method == 'umap':
    start = time.time()
    if supervised == False:
        embeddings = umap.UMAP(n_neighbors=neighbors,min_dist=min_dist,metric='correlation').fit_transform(hidden_embeddings)
    else:
        embeddings = umap.UMAP(n_neighbors=neighbors,min_dist=min_dist,metric='correlation').fit_transform(hidden_embeddings,labels)
    end = time.time()
    print(method,'time',end - start)

  elif method == 'ivis':
    start = time.time()
    if supervised == False:
        embeddings = Ivis(embedding_dims=2, k=neighbors).fit_transform(hidden_embeddings)
    else:
        embeddings = Ivis(embedding_dims=2, k=neighbors).fit_transform(hidden_embeddings, labels)
    end = time.time()
    print(method,'time',end - start)

  return embeddings

def print_statistics(embeddings,labels):
  silhouette_avg = silhouette_score(embeddings, labels)
  silhouette_std = np.std(silhouette_samples(embeddings, labels))
  print('silhouette average', silhouette_avg, 'std', silhouette_std)
  db_score = davies_bouldin_score(embeddings, labels)
  print('davies bouldin score',db_score)
  ch_score = calinski_harabasz_score(embeddings, labels)
  print('calinski harabasz score',ch_score)
  
def run_visualizations(method_list,hidden_embeddings,labels,perplexity=30,neighbors=50,min_dist=0.3,supervised=False):
  for method in method_list:
    print('running',method)
    embeddings = reduce_dims(hidden_embeddings,labels,method=method,perplexity=perplexity,neighbors=neighbors,min_dist=min_dist,supervised=supervised)
    plot_embeddings(embeddings,labels)