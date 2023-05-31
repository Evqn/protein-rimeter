from PIL import Image
from torchvision import transforms
import torch


def predict_model_single_image(model, image_path, label_encoder):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Same transformations as trainig
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    output = model(img)

    _, predicted_idx = torch.max(output, 1)

    # same label_encoder used when making dataset
    predicted_class = label_encoder.inverse_transform(predicted_idx.cpu().numpy())
    
    return predicted_class
