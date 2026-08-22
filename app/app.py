import os 
import sys 

# Adding project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import tempfile 

import numpy as np 
import faiss 
import torch 
import torchvision.models as models

import streamlit as st   

from src.preprocess import preprocess_image
from src.extract_embeddings import get_embedding 

# paths 
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)) 
)


INDEX_FILE = os.path.join(
    BASE_DIR, 
    "faiss_index",
    "index_1k.faiss"
)

METADATA_FILE = os.path.join(
    BASE_DIR, 
    "metadata",
    "image_paths.npy"
)


# Loading the model 
model = models.resnet50(weights="DEFAULT") 
model.fc = torch.nn.Identity() 
model.eval() 

# Loading the FAISS index 
index = faiss.read_index(INDEX_FILE) 

# Loading image paths 
image_paths = np.load(
    METADATA_FILE ,
    allow_pickle=True
)


#Streamlit UI things 
st.title ("Visual SEarch with FAISS (Facebook AI Similarity Search)")

st.write("Uplaod a proudct image to find similar images ") 

uploaded_file = st.file_uploader (
    "Upload an image to Search" ,
    type = ["jpg", "jpeg", "png"] 
)


# Search now 
if uploaded_file is not None :
    # Display the query image(uploaded one) 
    st.image(
        uploaded_file,
        caption = "Query image",
        width = 300
    )
    
    # save uploaded imahe temporarily 
    with tempfile.NamedTemporaryFile(
        delete= False ,
        suffix=".jpg" 
    ) as temp_file :
        temp_file.write(
            uploaded_file.getbuffer() 
        )
        temp_image_path = temp_file.name 
        
    
    # preprocess the uploaded image 
    image_tensor = preprocess_image(temp_image_path) 
    
    # adding batch domention 
    image_tensor = image_tensor.unsqueeze(0) 
    
    # generating embedding for the uploade query image 
    query_embedding = get_embedding(image_tensor) 
    
    # k = how many similar images user want 
    k = st.slider(
        "Number of similar images required : ",
        min_value=1,
        max_value=6,
        value=3
    )
    
    # Search using FAISS 
    distances , ids = index.search(
        query_embedding,
        k
    )
    
    # displaying the results 
    st.subheader ("Similar images :") 
    columns = st.columns(k) 
    
    for i in range(k)  :
        image_id = ids[0][i]
        image_path = image_paths[image_id] 
        
        # make the path to absolute path 
        image_path = os.path.normpath(
            os.path.join(
                BASE_DIR,
                "src",
                image_path
            )
        )       
    
        
        with columns[i] :
            st.image(
                image_path,
                caption = f"Dostance : {distances [0][i] :.2f}"
            )
    
    
    