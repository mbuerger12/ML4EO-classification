import configargparse
import os
parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', is_config_file=True, help='Path to the config file', type=str)

# general
parser.add_argument('--save-dir', default="./save/" , help='Path to directory where models and logs should be saved')
parser.add_argument('--logstep-train', default=10, type=int, help='Training log interval in steps')
parser.add_argument('--save-model', default='both', choices=['last', 'best', 'no', 'both'])
parser.add_argument('--val-every-n-epochs', type=int, default=5, help='Validation interval in epochs')
parser.add_argument('--wandb', action='store_true', default=False, help='Use Weights & Biases instead of TensorBoard')
parser.add_argument('--wandb-project', type=str, default='ML4EO', help='Wandb project name')
parser.add_argument('--num_epochs', default=1, help='number of epochs to train')
parser.add_argument('--batch-size', default=8, help='batch size')
parser.add_argument('--dataset',default="full", help='dataset name')
parser.add_argument('--lr', default=0.001, help='learning rate')
parser.add_argument('--image-size', help='model name')
parser.add_argument('--sampler', default='random', help='sampling strategy')
parser.add_argument('--model', default='segformer', choices=['segformer', 'resnet50', 'ownCNN', 'randomforest'], help='model name')



# Random Forest args
parser.add_argument('--rf-n-estimators', default=100, help='Number of trees in Random Forest')
parser.add_argument('--rf-max-depth', default=None, help='Maximum depth of trees in Random Forest')
parser.add_argument('--rf-random-state', default=42, help='Random seed for Random Forest')
parser.add_argument('--rf-class-weight', default='balanced', help='Class weights for Random Forest')


#Include Layer args
parser.add_argument('--layer', default='layer/Berlin', help='Layer name')
parser.add_argument('--use_layer', default=False, help='Use layer data in training')
