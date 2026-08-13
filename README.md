# 🔎 Visual Search & Vector Retrieval Engine with FAISS

An image-based product search system that retrieves visually similar products from an e-commerce image dataset. (Amazon Berkeley Objects (ABO) Dataset)

The project uses image embeddings generated from a pretrained ResNet50 model and FAISS for vector similarity search. A query image is converted into an embedding and compared against the indexed product embeddings to retrieve the most similar items.

## 🎯 Project Overview

Traditional product search generally relies on text, keywords, or structured attributes.

This project explores **visual search**, where the input itself is an image.

<div align="center">

<b>Visual Search Workflow</b>

<br><br>

🖼️
<br>
<b>Query Image</b>

⬇️

⚙️
<br>
<b>Image Preprocessing</b>

⬇️

🔍
<br>
<b>Feature Extraction</b>

⬇️

📊
<br>
<b>Image Embedding</b>

⬇️

⚡
<br>
<b>FAISS Vector Search</b>

⬇️

🎯
<br>
<b>Top-K Similar Products</b>

</div>

## 🛠️ Tech Stack

- Python
- PyTorch / Torchvision
- ResNet50
- NumPy
- FAISS
- FastAPI
- Streamlit
- Docker


## 📦 Dataset

The project uses product images from the **Amazon Berkeley Objects (ABO)** dataset.

The initial prototype uses a subset of approximately 1,000 product images rather than the complete dataset.

The project is structured so that the image collection can be increased later without changing the basic retrieval workflow.


## 🧠 Embedding Generation

A pretrained ResNet50 model is used as a feature extractor.

The final classification layer is removed because the objective is not image classification. The resulting feature representation is used as the image embedding.

For the current prototype:

- Images processed: 999
- Embedding dimension: 2048
- Data type: `float32`
- Embedding storage: NumPy `.npy` file


## 🔍 Vector Retrieval

FAISS is used to index the generated image embeddings.

For a query image:

1. Preprocess the image.
2. Generate its embedding.
3. Search the FAISS index.
4. Retrieve the nearest vectors.
5. Map the retrieved IDs back to the corresponding products.
6. Display the most similar products.


## 📊 Current Progress

- [x] Dataset preparation
- [x] Image preprocessing
- [x] ResNet50 feature extraction
- [x] Batch embedding generation
- [x] 999 image embeddings generated
- [ ] FAISS index
- [ ] Similarity search
- [ ] Image ID → product mapping
- [ ] Query image retrieval
- [ ] FastAPI API
- [ ] Streamlit interface
- [ ] Docker setup


## 📁 Project Structure

```text
Visual Search & Vector Retrieval Engine/
│
├── data/
│   └── images/
│
├── embeddings/
│   └── embeddings_1k.npy
│
├── src/
│   ├── preprocess.py
│   ├── extract_embeddings.py
│   └── ...
│
├── requirements.txt
├── README.md
└── .gitignore
