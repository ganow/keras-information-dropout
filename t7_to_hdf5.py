import torchfile
import h5py

dataset_types = ('train', 'valid', 'test')
dataset_path = 'data/cluttered_{}.t7'
outpath = 'data/cluttered_mnist.h5'

with h5py.File(outpath, 'w') as hf:

  for dataset_type in dataset_types:
    inpath = dataset_path.format(dataset_type)
    print('... load {}'.format(inpath))
    o = torchfile.load(inpath)

    print('... save {}, shape: {}'.format('X_{}'.format(dataset_type), o[b'data'].shape))
    hf.create_dataset('X_{}'.format(dataset_type), data=o[b'data'])
    print('... save {}, shape: {}'.format('Y_{}'.format(dataset_type), o[b'labels'].shape))
    hf.create_dataset('Y_{}'.format(dataset_type), data=o[b'labels'])
