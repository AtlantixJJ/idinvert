# python 3.6
import os
import argparse
import sys
sys.path.insert(0, ".")
import pickle
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from dnnlib import tflib

from utils.visualizer import adjust_pixel_range, save_image, VideoWriter


def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('model_path', type=str,
                      help='Path to the pre-trained model.')
  parser.add_argument('src_dir', type=str,
                      help='Source directory, which includes original images, '
                           'inverted codes, and image list.')
  parser.add_argument('--gpu_id', type=str, default='0',
                      help='Which GPU(s) to use. (default: `0`)')
  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
  # Load model.
  tflib.init_tf({'rnd.np_random_seed': 1000})
  with open(args.model_path, 'rb') as f:
    _, _, _, Gs = pickle.load(f)

  # Build graph.
  batch_size = 40
  sess = tf.get_default_session()
  num_layers, latent_dim = Gs.components.synthesis.input_shape[1:3]
  wp = tf.placeholder(
      tf.float32, [batch_size, num_layers, latent_dim], name='latent_code')
  x = Gs.components.synthesis.get_output_for(wp, randomize_noise=False)

  # Load image and codes.
  encoded_codes = np.load(f'{args.src_dir}/encoded_codes.npy')
  optimized_codes = np.load(f'{args.src_dir}/inverted_codes.npy')
  print(encoded_codes.shape, optimized_codes.shape)
  output_dir = args.src_dir

  DNUM = 10
  DSIZE = 80

  for img_idx in tqdm(range(16)): # image index
    rem_video = VideoWriter(f'{args.src_dir}/rem_{img_idx}.mp4', 256, 256)

    start = img_idx * DNUM * DSIZE
    original_code = optimized_codes[img_idx]
    if optimized_codes.shape[0] > 16:
      original_code = optimized_codes[img_idx * 10]
    for d_idx in range(10): # direction index
      codes = encoded_codes[start + d_idx * DSIZE:][:DSIZE]
      newcodes = codes.copy()
      newcodes[0] = original_code
      newcodes[1:] = codes[1:] - codes[0] + original_code

      output_images = []
      for idx in range(len(newcodes) // batch_size):
        images = sess.run(x, feed_dict={wp: newcodes[idx * batch_size : idx * batch_size + batch_size]})
        output_images.append(images)
      output_images = adjust_pixel_range(np.concatenate(output_images, axis=0))
      for i in range(len(output_images)):
        #save_image(f'{output_dir}/{img_idx:06d}_transform{d_idx*DSIZE+i:03d}_rem.png', output_images[i])
        rem_video.write(output_images[i])

    del rem_video


if __name__ == '__main__':
  main()
