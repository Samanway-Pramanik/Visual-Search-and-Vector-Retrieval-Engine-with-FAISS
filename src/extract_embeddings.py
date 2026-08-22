import os
import numpy as np 

import torch
import torchvision.models as models

from torch.utils.data import Dataset, DataLoader 

# import the function from preprocess.py
try:
    from src.preprocess import preprocess_image
except ModuleNotFoundError:
    from preprocess import preprocess_image


# Loading the pretrained ResNet50 model 

model = models.resnet50(weights="DEFAULT") 
model.fc = torch.nn.Identity()   # we don't need the final classification layer, so removing it 
model.eval() 


# Dataset --------------------------------
class ImageDataset (Dataset) :
    def __init__(self, image_paths)  :
        self.image_paths = image_paths 
        
    def __len__(self) :
        return len(self.image_paths) 
    
    def __getitem__(self, index) :
        image_path = self.image_paths[index] 
        image = preprocess_image(image_path)  
        
        return image 
    

# The Embedding func -------------------------
def get_embedding (images) :
    
    with torch.no_grad() :
        embedding = model(images) 
        embedding = embedding.cpu().numpy().astype("float32") 
        # L2 normalisation for cosine similarity 
        embedding = embedding / np.linalg.norm(
            embedding,
            axis = 1,
            keepdims=True
        )
    return embedding 


# Main function ------------------------------
if __name__ == "__main__" :
    image_dir = "../data/images/images/small/00" 
    output_file = "../embeddings/embeddings_1k.npy" 
    
    paths = []   # for storing valid image paths 
    
    for filename in os.listdir(image_dir) :
        if filename.lower().endswith( (".jpg", ".png") ) :
            path = os.path.join(image_dir, filename) 
            paths.append(path) 
            

    dataset = ImageDataset(paths) 
    
    loader = DataLoader(
        dataset,
        batch_size = 32,
        shuffle = False
    )
    
    all_embeddings = []             # to store embeddings 
    
    for images in loader :
        batch_embeddings = get_embedding(images)      # embeddings of whole one batch
        all_embeddings.append(batch_embeddings) 
        
        
    # combining embedings from all batches 
    all_embeddings = np.vstack(all_embeddings)    # stack all the batch embeddings vertically
    
    # save the embeddings 
    os.makedirs("../embeddings", exist_ok=True) 
    np.save(output_file, all_embeddings)
    
    print("Embedding shape :" ,all_embeddings.shape ) 
    print("Saved to ... : ", output_file ) 
    
    os.makedirs("../metadata", exist_ok=True)
    np.save("../metadata/image_paths.npy", np.array(paths))  # we need imahe paths , as FAISS
                                                             # will return only IDs  