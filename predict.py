import argparse
from inference import inference

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(action='store', dest='image_dir', help='Image directory')
    parser.add_argument(action='store', dest='checkpoint_dir', help='Checkpoint directory')
    parser.add_argument('--top_k', type=int, action='store', dest='top_k', help='Top K most likely classes',
                        default=5)
    parser.add_argument('--category_names', action='store', dest='category_names', help='Mapping of categories to real names', 
                        default='cat_to_name.json')
    parser.add_argument('--gpu', type=bool, action='store', dest='gpu', help='GPU Available',
                        default=False)
    
    results = parser.parse_args()
    print('Image directory       = {}'.format(results.image_dir))
    print('Checkpoint directory  = {}'.format(results.checkpoint_dir))
    print('Top K                 = {}'.format(results.top_k))
    print('Category Names        = {}'.format(results.category_names))
    print('GPU Available         = {}'.format(results.gpu))
    
    image_to_predict = inference(results.image_dir, results.checkpoint_dir, results.top_k, results.category_names, results.gpu)
    
    print('\nPrediction with the following model:')
    image_to_predict.load_checkpoint()
    
    print('\nResult:')
    image_to_predict.predict()
    
 
if __name__ == "__main__":
    main()
