import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json

class inference:
    def __init__(self, image_dir, checkpoint_dir, top_k, category_names, gpu):
        self.image_dir =image_dir
        self.checkpoint_dir = checkpoint_dir
        self.top_k = top_k
        self.category_names = category_names
        self.gpu = gpu
        
    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_dir)
        
        if checkpoint['structure'] == 'vgg16':
            self.model = models.vgg16(pretrained=True)
        elif checkpoint['structure'] == 'densenet121' :
            self.model = models.densenet121(pretrained=True)    
        
        self.model.class_to_idx = checkpoint['class_to_idx']
        
        self.optimizer = checkpoint['optimizer_state_dict']
        #self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.model.classifier = checkpoint['classifier']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        
        epoch = checkpoint['epoch']
        
        print(checkpoint['structure'])
        print(self.model.classifier)
        
        
    def process_image(image_path):
        image_PIL = Image.open(image_path)
        img_transforms = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])])
        img_transform = img_transforms(image_PIL)
        
        return img_transform
        
        
    def predict(self):
        
        # Use GPU if it's available
        device = torch.device("cuda" if self.gpu else "cpu")
        
        # Find the top K
        imgage_new = inference.process_image(self.image_dir)
        imgage_new = imgage_new.unsqueeze_(0)
        self.model.to(device)
        
        self.model.eval()
        
        with torch.no_grad():
            imgage_new = imgage_new.to(device)
            outputs = self.model.forward(imgage_new)
        
        ps = F.softmax(outputs.data, dim=1)
        top_p, top_class = ps.topk(self.top_k, dim=1)
        
        class_idx_dict = {self.model.class_to_idx[key]: key for key in self.model.class_to_idx}
        
        classes = []
        
        cpu_labels = top_class.cpu()
        for label in cpu_labels.detach().numpy()[0]:
            classes.append(class_idx_dict[label])
            
        # Label mapping
        with open(self.category_names, 'r') as f:
            cat_to_name = json.load(f)
            
        flower_names = [cat_to_name[i] for i in classes]
        
        # Print the result
        proba = top_p.cpu().numpy()[0][0] * 100
        print('\nPredicted flower name: {} with a probability of: {:.2f}%\n'.format(flower_names[0], proba))
        print('The top {} most likely classes are:\n'.format(self.top_k))
        
        for i in range(len(flower_names)):
            proba = top_p.cpu().numpy()[0][i] * 100
            print('{} with {:.2f}%'.format(flower_names[i], proba))


