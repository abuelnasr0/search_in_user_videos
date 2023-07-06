import faiss
import numpy as np 

class VDB():
    def __init__(self, num_entities, cell_size, embedding_dim=768):
        num_centroids = num_entities// cell_size 
        quantizer = faiss.IndexFlatIP(embedding_dim)
        self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, num_centroids)
        self.num_entities = num_entities
        self.num_centroids = num_centroids
        self.captions = None

    def train(self, embeddings):
        #test if its trained
        self.index.train(embeddings)

    def insert(self, embeddings, captions):
        #test if num_entities == ntotal in the index
        self.index.add(embeddings)
        self.captions = captions

    def search(self, embeddings, top_k):
        if len(embeddings.shape) == 1 :
            embeddings = np.expand_dims(embeddings, 0)
        distances, matches = self.index.search(embeddings, top_k)
        final_results = []
        for match in matches:
            result = []
            for index in match:
                if index == -1:
                    break
                result.append(self.captions[index])
            final_results.append(result)

        return final_results

