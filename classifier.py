import ast
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import torch

# Load pre-trained models
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

model_dict = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}

# Obtain ImageNet labels
with open('imagenet1000_clsid_to_human.txt') as imagenet_classes_file:
    imagenet_classes_dict = ast.literal_eval(imagenet_classes_file.read())

def load_image(img_path):
    try:
        img_pil = Image.open(img_path)
        print(f"Successfully loaded image: {img_path}")
        return img_pil
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None

def preprocess_image(img_pil):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = preprocess(img_pil)
    img_tensor.unsqueeze_(0)  # Add dimension for batch
    print(f"Preprocessed image to tensor: {img_tensor.shape}")
    return img_tensor

def classify_image(img_tensor, model_name):
    model = model_dict[model_name]
    model.eval()
    
    with torch.no_grad():
        output = model(img_tensor)
    
    print(f"Model output: {output}")
    
    # Convert output tensor to numpy array and get the predicted label index
    pred_idx = output.numpy().argmax()
    predicted_label = imagenet_classes_dict[pred_idx]
    print(f"Predicted label index: {pred_idx}, label: {predicted_label}")
    return predicted_label

def classifier(img_path, model_name):
    img_pil = load_image(img_path)
    if img_pil is None:
        return ""
    
    img_tensor = preprocess_image(img_pil)
    predicted_label = classify_image(img_tensor, model_name)
    return predicted_label
