import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

class classifier:
    def __init__(self, images_dir, results_dic, architecture, learning_rate, hidden_units, epochs, gpu):
        self.images_dir = images_dir
        self.results_dic = results_dic
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.gpu = gpu
        
        self.batch_size = 64
        
        
    def load_image(self):
        data_dir = self.images_dir
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'
        
        #Define the transforms for the training, validation, and testing sets
        train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                               transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        valid_transforms = transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        test_transforms = transforms.Compose([transforms.Resize(255),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        # Load the datasets with ImageFolder
        self.train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
        self.valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)
        self.test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms )

        # Define the dataloaders
        self.trainloader = torch.utils.data.DataLoader(self.train_datasets, batch_size = self.batch_size, shuffle = True)
        self.validloader = torch.utils.data.DataLoader(self.valid_datasets, batch_size = self.batch_size)
        self.testloader = torch.utils.data.DataLoader(self.test_datasets, batch_size = self.batch_size)

        return test_dir
    
    
    def load_model(self):
        if self.architecture == 'vgg16':
            self.model = models.vgg16(pretrained=True)
        elif self.architecture == 'densenet121' :
            self.model = models.densenet121(pretrained=True)
        else:
            print('{} architecture not available, vgg16 is used by default'.format(self.architecture))
            self.model = models.vgg16(pretrained=True)
        return self.model
    
    def train_model(self):
        print_every = 30
        steps = 0
        
        # Use GPU if it's available
        device = torch.device("cuda" if self.gpu else "cpu")
        
        # Freeze the parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # define a new classifier
        classifier = nn.Sequential(nn.Linear(self.hidden_units[0],self.hidden_units[1]),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),
                                   nn.Linear(self.hidden_units[1],102),
                                   nn.LogSoftmax(dim = 1))
        
        self.model.classifier = classifier
        
        # Define loss and optimizer
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr = self.learning_rate)
        self.model.to(device)
        
        # Train the pretrained model
        running_loss = 0
        for epoch in range(self.epochs):
            for inputs, labels in self.trainloader:
                steps += 1
                inputs, labels = inputs.to(device), labels.to(device)
                
                self.optimizer.zero_grad()
                log_ps = self.model.forward(inputs)
                loss = self.criterion(log_ps, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                
                # Test the network accuracy
                if steps % print_every == 0:
                    valid_loss = 0
                    accuracy = 0
                    self.model.eval()
                    with torch.no_grad():
                        for inputs, labels in self.validloader:
                            inputs, labels = inputs.to('cuda'), labels.to('cuda')
                            
                            log_ps = self.model.forward(inputs)
                            loss = self.criterion(log_ps, labels)
                            valid_loss += loss.item()
                            
                            # Accuracy
                            ps = torch.exp(log_ps)
                            top_p, top_class = ps.topk(1, dim = 1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                    print('Epoch: {}/{} ... '.format(epoch +1, self.epochs),
                    'Train loss: {:.3f} ... '.format(running_loss / (print_every)),
                    'Valid loss: {:.3f} ... '.format(valid_loss / len(self.validloader)),
                    'Valid accuracy: {:.3f} ... '.format(accuracy / len(self.validloader)))
                    self.model.train()
                    running_loss = 0
                    

    def test_model(self):
        test_loss = 0
        test_accuracy = 0
        
        # Use GPU if it's available
        device = torch.device("cuda" if self.gpu else "cpu")
        
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                log_ps = self.model.forward(inputs)
                loss = self.criterion(log_ps, labels)
                test_loss += loss.item()
                
                # Accuracy
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim = 1)
                equals = top_class == labels.view(*top_class.shape)
                test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        print('Test loss: {:.3f} ... '.format(test_loss / len(self.testloader)),
              'Test accuracy: {:.3f} ... '.format(test_accuracy / len(self.testloader)))
        self.model.train()

    def save_model(self):
        self.model.class_to_idx = self.train_datasets.class_to_idx
        checkpoint = {'input_size': self.hidden_units[0],
                      'output_size': 102,
                      'structure': self.architecture,
                      'learning_rate': self.learning_rate,
                      'classifier': self.model.classifier,
                      'epoch': self.epochs,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'class_to_idx': self.model.class_to_idx}
        
        torch.save(checkpoint, self.results_dic)


        


    