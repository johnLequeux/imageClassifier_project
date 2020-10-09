import argparse
from classifier import classifier

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(action='store', dest='data_dir', help='Data directory')
    parser.add_argument('--save_dir', action='store', dest='save_dir', help='Save directory', 
                        default='checkpoint.pth')
    parser.add_argument('--arch', action='store', dest='architecture', help='Architecture (vgg16 or densenet121)', 
                        default='vgg16')
    parser.add_argument('--learning_rate', type=float, action='store', dest='learning_rate', help='Learning Rate', 
                        default=0.001)
    parser.add_argument("--hidden_units", type=int, nargs="+", dest='hidden_units', help='List of hidden Unit', 
                        default=[25088,4096])
    parser.add_argument('--epochs', type=int, action='store', dest='epochs', help='Epochs',
                        default=3)
    parser.add_argument('--gpu', type=bool, action='store', dest='gpu', help='GPU Available',
                        default=False)
    
    results = parser.parse_args()
    print('Data directory     = {}'.format(results.data_dir))
    print('Save directory     = {}'.format(results.save_dir))
    print('Architecture       = {}'.format(results.architecture))
    print('Learning Rate      = {}'.format(results.learning_rate))
    print('Hidden Unit        = {}'.format(results.hidden_units))
    print('Epochs             = {}'.format(results.epochs))
    print('GPU Available      = {}'.format(results.gpu))
    
    model_to_train = classifier(results.data_dir, results.save_dir, results.architecture, 
                                results.learning_rate, results.hidden_units, results.epochs, results.gpu)
    
    print('\nTrain the classifier:')
    model_to_train.load_image()
    model_to_train.load_model()
    model_to_train.train_model()
    
    print('\nTest the classifier:')
    model_to_train.test_model()
    model_to_train.save_model()
    
    print('\nCheck point saved in: {}'.format(results.save_dir))
    
    '''
    print("\ntest1: " + model_to_train.architecture)
    print("test2: " + model_to_train.load_image())
    print(model_to_train.load_model())
    '''
    
if __name__ == "__main__":
    main()
