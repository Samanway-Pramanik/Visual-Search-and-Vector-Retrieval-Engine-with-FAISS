# 🔎 Visual Search & Vector Retrieval Engine with FAISS

An image-based product search system that retrieves visually similar products from an e-commerce image dataset. (Amazon Berkeley Objects (ABO) Dataset)

The project uses image embeddings generated from a pretrained ResNet50 model and FAISS for vector similarity search. A query image is converted into an embedding and compared against the indexed product embeddings to retrieve the most similar items.

## 🎯 Problem & Business Value

Problem:
Traditional e-commerce search relies heavily on text-based queries, which can make it difficult for users to find products when they don't know the exact product name, keywords, or description. This creates friction in visual product discovery, especially for categories such as fashion, footwear, furniture, and accessories.

Business Value:
This project enables image-based product discovery by allowing users to upload an image and retrieve visually similar products from a large catalog. This can improve the shopping experience, reduce search friction, and provide the foundation for business use cases such as visual search, similar-product recommendations, product discovery, and cross-selling.

## 🎯 Project Overview

Traditional product search generally relies on text, keywords, or structured attributes.

This project explores **visual search**, where the input itself is an image.

<div align="center">
 
<b>Visual Search Workflow</b>

<br><br>
🖼️
<br>
<b>Query Image
</b>
⬇️
<br>
<b>Image Preprocessing
</b>
⬇️
<br>
<b>Feature Extraction
</b>
⬇️
<br>
<b>Image Embedding
</b>
⬇️
<br>
<b>FAISS Vector Search
</b>
⬇️
<br>
<b>Top-K Similar Products
</b>
</div>

## 🛠️ Tech Stack

- Python
- PyTorch
- Torchvision
- ResNet50
- NumPy
- FAISS
- Streamlit

---

## 📦 Dataset

The project uses product images from the **Amazon Berkeley Objects (ABO) Dataset**.

For this prototype, a subset of approximately **1,800 product images** is used instead of the complete dataset.

The pipeline is designed so that the image collection can be increased later without changing the overall retrieval workflow.

---

## 🧠 Image Embedding Generation

A pretrained **ResNet50** model is used as the feature extractor.

The final classification layer is removed because the objective is not image classification. Instead, the resulting **2048-dimensional feature representation** is used as the image embedding.

For each image, the following process is performed:

**Product Image → Preprocessing → ResNet50 → 2048-D Embedding**

### Current Configuration

- **Images indexed:** ~1,800
- **Embedding dimension:** 2048
- **Data type:** `float32`
- **Embedding storage:** NumPy `.npy`

---

## 🔍 Vector Retrieval with FAISS

**FAISS (Facebook AI Similarity Search)** is used to perform similarity search over the generated image embeddings.

For a query image:

1. The user uploads an image through the Streamlit interface.
2. The image is temporarily saved.
3. The image is preprocessed using the same preprocessing pipeline used for the dataset images.
4. ResNet50 generates a **2048-dimensional embedding** for the query image.
5. The query embedding is passed to the FAISS index.
6. FAISS searches for the nearest vectors in the indexed embedding collection.
7. The returned vector IDs are mapped back to the corresponding image paths using the stored metadata.
8. The most visually similar product images are displayed to the user.

The number of retrieved results can be selected through the Streamlit interface.

---

## 🖥️ Streamlit Application

The project includes an interactive **Streamlit** interface for performing visual product search.

Users can:

- Upload a product image
- Select the number of similar images to retrieve
- View the uploaded query image
- View the retrieved visually similar products
- See the corresponding FAISS similarity distances

The application performs the complete retrieval pipeline at inference time:

**Query Image → Preprocessing → ResNet50 → Embedding → FAISS Search → Similar Images**

---

## 📊 Current Status

- [x] Dataset preparation
- [x] Image subset extraction
- [x] Image preprocessing
- [x] ResNet50 feature extraction
- [x] Batch embedding generation
- [x] FAISS index creation
- [x] Image ID → image path mapping
- [x] Similarity search
- [x] Query image retrieval
- [x] Streamlit interface

### Future Improvements

- [ ] Increase the indexed image collection
- [ ] Improve embedding quality
- [ ] Experiment with different similarity metrics
- [ ] Add product metadata to search results
- [ ] Optimize inference and retrieval performance
- [ ] Deploy the application publicly

---

## 📁 Project Structure

```text
Visual Search & Vector Retrieval Engine/
│
├── app/
│   └── app.py
│
├── data/
│   └── images/
│
├── embeddings/
│   └── embeddings_1k.npy
│
├── faiss_index/
│   └── index_1k.faiss
│
├── metadata/
│   └── image_paths.npy
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── extract_subset.py
│   ├── extract_embeddings.py
│   ├── build_index.py
│   └── search.py
│
├── requirements.txt
├── .gitignore
└── README.md
