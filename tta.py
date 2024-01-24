import torch
import torch.nn.functional as F
from torchvision import transforms as T


def ttaug(model, output_orig, img, mask):
    transforms = []
    transforms.append(T.Compose([T.RandomRotation((-30, 30)), T.Resize((224, 224))]))
    transforms.append(T.RandomCrop((224, 224)))
    transforms.append(T.Compose([T.RandomHorizontalFlip(1), T.Resize((224, 224))]))
    transforms.append(T.Compose([T.RandomRotation((-30, 30)), T.RandomCrop((224, 224))]))
    transforms.append(T.Compose([T.RandomRotation((-30, 30)), T.RandomHorizontalFlip(1), T.Resize((224, 224))]))
    transforms.append(T.Compose([T.RandomHorizontalFlip(1), T.RandomCrop((224, 224))]))
    transforms.append(T.Compose([T.RandomRotation((-30, 30)), T.RandomHorizontalFlip(1), T.RandomCrop((224, 224))]))
    mask_size = torch.sum(mask).item()
    outputs = [output_orig]
    for transformer in transforms:
        i = 0
        while i < 10:
            augmented_image = transformer(img)
            augmented_mask = transformer(mask)
            if torch.sum(augmented_mask.squeeze()).item() > mask_size-10:
                model_output = F.softmax(model(augmented_image), dim=-1).data
                outputs.append(model_output)
                break
            i += 1
    outputs = torch.stack(outputs, dim=1)

    return outputs