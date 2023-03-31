import torch
from torchvision.datasets import ImageNet
from torchvision.transforms import transforms
from models.model_load import model_load
from PIL import Image 

if __name__ == "__main__":
    device = torch.device("cpu")
    model = model_load("inceptionv3")

    class_dir = ImageNet("F:\\ImageNet",download=False).class_to_idx
    id_to_class_dir = {value:key for (key,value) in class_dir.items()}

    model.eval()
    model.to(device=device)

    crop = transforms.Compose([
        transforms.Resize([299,299]),
        transforms.ToTensor()
    ])

    img = Image.open("F:\\hierarchical_concept_explanation\\zebra\\1_Zebra1.jpg")
    img = img.convert('RGB')
    img = crop(img).unsqueeze(0)

    with torch.no_grad():
        prediction = model(img)
        result_id = torch.argmax(torch.softmax(prediction,dim=1))
        print(id_to_class_dir[result_id.item()])

    
