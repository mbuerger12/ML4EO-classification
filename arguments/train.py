import configargparse

parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', is_config_file=True, help='Path to the config file', type=str)

# general
parser.add_argument('--save-dir', default="./save/" , help='Path to directory where models and logs should be saved')
parser.add_argument('--logstep-train', default=10, type=int, help='Training log interval in steps')
parser.add_argument('--save-model', default='both', choices=['last', 'best', 'no', 'both'])
parser.add_argument('--val-every-n-epochs', type=int, default=1, help='Validation interval in epochs')
parser.add_argument('--wandb', action='store_true', default=False, help='Use Weights & Biases instead of TensorBoard')
parser.add_argument('--wandb-project', type=str, default='ML4EO', help='Wandb project name')
parser.add_argument('--num_epochs', default=100, help='number of epochs to train')
parser.add_argument('--batch-size', default=8, help='batch size')
parser.add_argument('--dataset',default="berlin", help='dataset name')
parser.add_argument('--lr', default=0.001, help='learning rate')
parser.add_argument('--image-size', help='model name')