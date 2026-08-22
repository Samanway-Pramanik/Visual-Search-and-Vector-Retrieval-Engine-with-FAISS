import numpy as np 
import faiss

import torch 
import torchvision.models as models

# Loading the pretrained Resnet50 model ----------
model = models.resnet50(weights= "DEFAULT") 
model.fc = torch.nn.Identity()    # remocing the classification layer 
model.eval() 

# Embedding functin 

from preprocess import preprocess_image   # importing the function from preprocess.py

def get_embedding(image_path) :
    image = preprocess_image(image_path) 
    image = image.unsqueeze(0) 
    
    with torch.no_grad() :
        embedding = model(image) 
    
    embedding = embedding.squeeze().cpu().numpy().astype("float32") 
    # L2 normalisation for cosine similarity 
    embedding = embedding / np.linalg.norm(embedding)
    return embedding 


# Loading the FAISS indexes 
index_file = "../faiss_index/index_1k.faiss" 
index = faiss.read_index(index_file) 

print("FAISS index loaded ." )
print(f"Number of vectors : {index.ntotal}") 


# Load the image paths 
image_paths = np.load(
    "../metadata/image_paths.npy",
    allow_pickle=True
)


# Query Image , Testing mannually , later through streamlit, user will upload an image.....
query_image = "../data/images/images/small/00/00a3f79e.jpg"
query_embedding = get_embedding(query_image) 
 
# Search for top k similar images 
k = 5

distances , ids = index.search(
    query_embedding.reshape(1,-1),
    k
)


# Showing the result 
print("\n similar images after search : ") 

for i in range(k):
    image_id = ids[0][i] 
    distance = distances[0][i] 
    image_path = image_paths[image_id] 
    
    print(
        f"{i+1}. "
        f"ID : {image_id} | "
        f"Distance = {distance:.2f} | "
        f"Image : {image_path}"
    )
