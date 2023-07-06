from sentence_transformers import SentenceTransformer
import torch

model_name = 'msmarco-bert-base-dot-v5'
model = SentenceTransformer(model_name) 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
