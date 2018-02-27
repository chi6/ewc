import argparse

def str2bool(v):
    return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--n_classes', type=int, default=100)  # Number of classes

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='cifar100')
data_arg.add_argument('--batch_size', type=int, default=10)

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--optimizer', type=str, default='adam')
train_arg.add_argument('--skip_step', type=int, default=10)
train_arg.add_argument('--epoch_step', type=int, default=1)
train_arg.add_argument('--learning_rate', type=float, default=1e-3)
train_arg.add_argument('--dropout', type=float, default=0.75)
train_arg.add_argument('--l2_lambda', type=float, default=0)

# Misc
misc_arg = add_argument_group('Misc')


def get_config():
  config, unparsed = parser.parse_known_args()
  print(config)
  return config, unparsed

if __name__ == '__main__':

    get_config()