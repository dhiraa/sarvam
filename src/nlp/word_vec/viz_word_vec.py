from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
sys.path.append("../")
import word_vec.utils.common as common

EMBED_MAT_PATH = "tmp/embed_mat.npy"
VOCAB_PATH = "tmp/vocab.tsv"

# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)


def visulize_word_vec(final_embeddings=np.load("tmp/embed_mat.npy"), vocab_path="tmp/vocab.tsv", save_path="tmp/"):
    word_2_id, id_2_word = common.tsv_to_vocab("tmp/vocab.tsv")
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [id_2_word[i] for i in range(plot_only)]
    plot_with_labels(low_dim_embs, labels, os.path.join(save_path, 'word_cloud.png'))

if __name__ == "__main__":

    if not os.path.exists(EMBED_MAT_PATH) or not os.path.exists(VOCAB_PATH):
        print("Train the model, before you can visulaize with it!!!")
        exit(0)

    visulize_word_vec()

