import os
import numpy as np
import h5py
from PIL import Image

IMG_SIZE = (96, 96)
padding = 2
N_width = 4
N_height = 4
CANVAS_SIZE = (
  IMG_SIZE[0]*N_width + padding*(N_width+1),
  IMG_SIZE[1]*N_height + padding*(N_height+1)
)

inpath = 'data/cluttered_mnist.h5'
filepath = 'images/datasample.png'

if not os.path.isdir('images'):
  os.mkdir(images)

with h5py.File(inpath, 'r') as hf:

  X = hf.get('X_train')
  labels = hf.get('Y_train')

  idxs = np.random.choice(len(X), size=N_width*N_height, replace=False)

  canvas = Image.new('L', size=CANVAS_SIZE, color='white')
  for i in range(N_width):
    for j in range(N_height):
      idx = idxs[i*N_height+j]
      img = Image.fromarray(X[idx, :, :], mode='L')
      canvas.paste(
        img, (
          IMG_SIZE[0]*i + padding*(i+1),
          IMG_SIZE[1]*j + padding*(j+1)
        )
      )
  canvas.save(filepath, 'png')
