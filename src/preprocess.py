from PIL import Image 
from torchvision import transforms

# preprocessing 
preprocess = transforms.Compose(
    [
        transforms.Resize(256) ,
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225] 
        )
    ]
)

def preprocess_image(image_path) :
    image = Image.open(image_path).convert("RGB") 
    image_tensor = preprocess(image) 
    
    return image_tensor 



# test 

if __name__ == "__main__" :
    image_path = "/home/kiit/Career/ML/projects/Visual Search with FAISS/data/images/images/small/00/00a0a32d.jpg"
    image_tensor = preprocess_image(image_path) 

    print(image_tensor.shape)