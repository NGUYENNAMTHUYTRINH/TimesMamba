import os
root='.'
dataset='weather'
candidates=[
    os.path.join(root, 'train', 'datasets', dataset, f'{dataset}_train.csv'),
    os.path.join(root, 'dataset', dataset, f'{dataset}.csv'),
    os.path.join(root, 'dataset', f'{dataset}.csv'),
    os.path.join(root, 'dataset', dataset, f'{dataset}_train.csv'),
]
for p in candidates:
    print(p, '->', os.path.exists(p))
