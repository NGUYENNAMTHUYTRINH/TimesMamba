import sys, os
root = os.getcwd()
sys.path.insert(0, root)
# Ensure package-local imports in subfolders work when running from project root
sys.path.insert(0, os.path.join(root, 'RNN'))
sys.path.insert(0, os.path.join(root, 'ITransformer'))
from RNN.train_rnn import Args
from data_provider.data_factory import data_provider

args = Args('weather')
print('Root path:', args.root_path)
print('Data path:', args.data_path)
try:
    data_set, loader = data_provider(args, 'train')
    print('Loaded dataset, length:', len(data_set))
    x,y,_,_ = data_set[0]
    print('Sample shapes:', x.shape, y.shape)
except Exception as e:
    print('Error loading dataset:', e)
