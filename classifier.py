import ast
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
import torch 

resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

model_dict = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}

# obtain ImageNet labels
with open('imagenet1000_clsid_to_human.txt') as imagenet_classes_file:
    imagenet_classes_dict = ast.literal_eval(imagenet_classes_file.read())
def load_image(img_path):
    try:
      img_pil = Image.open(img_path)
    except Exception as e:
      print(f"Error loading image {img_path}: {e}")
      return None

def preprocess_image(img_pil):
    # load the image
     preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # define transforms
    # preprocess the image
     img_tensor = preprocess(img_pil)
    
    # resize the tensor (add dimension for batch)
     img_tensor.unsqueeze_(0)
     return img_tensor
   
    
def classify_image(img_tensor, model_name):
    model=model_dict[model_name]
    model.eval()

    with torch.no_grad():
        output=model(img_tensor)

    pred_idx=output.data.numpy().argmax()
    predicted_label=imagenet_classes_dict[pred_idx]
    return predicted_label


def classifier(img_path,model_name):
    img_pil=load_image(img_path)
    if img_pil is None:
        return ""
    
    img_tensor = preprocess_image(img_pil)
    predicted_label=classify_image(img_tensor,model_name)
    return predicted_label
