# python 3.6
import os
import argparse
import pickle
from tqdm import tqdm
import sys
sys.path.insert(0, ".")
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
  parser.add_argument('--gpu-id', type=str, default='0',
                      help='Which GPU(s) to use. (default: `0`)')
  parser.add_argument('--mix-type', type=str, default='before',
                      help='before | after | number')
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

  t = "mix" if args.mix_type == 'before' else "mixafter"
  try:
    number = 45#int(args.mix_type)
    t = f"at{number}"
  except:
    pass
  
  for img_idx in tqdm(range(16)): # image index
    start = img_idx * DNUM * DSIZE
    original_code = optimized_codes[img_idx]
    if optimized_codes.shape[0] > 16:
      original_code = optimized_codes[img_idx * 10]

    if "at" in t:
      mix_video = VideoWriter(f'{args.src_dir}/{t}_{img_idx}.mp4', 256, 256)
      newcodes = np.repeat(np.expand_dims(original_code, 0), DSIZE, 0)

      for d_idx in range(10): # direction index
        codes = encoded_codes[start + d_idx * DSIZE:][:DSIZE]
        newcodes[:, 4] = codes[:, 4]
        newcodes[:, 5] = codes[:, 5]

        output_images = []
        for idx in range(DSIZE // batch_size):
          images = sess.run(x, feed_dict={wp: newcodes[idx * batch_size : idx * batch_size + batch_size]})
          output_images.append(images)
        output_images = adjust_pixel_range(np.concatenate(output_images, axis=0))
        for i in range(len(output_images)):
          #save_image(f'{output_dir}/{img_idx:06d}_transform{d_idx*DSIZE+i:03d}_rem.png', output_images[i])
          mix_video.write(output_images[i])

      del mix_video
      continue

    for mix_layer in [1, 2, 3, 4, 6, 8, 10, 12]:
      mix_video = VideoWriter(f'{args.src_dir}/{t}_{img_idx}_mix{mix_layer}.mp4', 256, 256)

      for d_idx in range(10): # direction index
        codes = encoded_codes[start + d_idx * DSIZE:][:DSIZE]
        newcodes = codes.copy()
        if args.mix_type == 'before':
          newcodes[:, mix_layer:] = original_code[mix_layer:]
        elif args.mix_type == 'after':
          newcodes[:, :mix_layer] = original_code[:mix_layer]

        output_images = []
        for idx in range(DSIZE // batch_size):
          images = sess.run(x, feed_dict={wp: newcodes[idx * batch_size : idx * batch_size + batch_size]})
          output_images.append(images)
        output_images = adjust_pixel_range(np.concatenate(output_images, axis=0))
        for i in range(len(output_images)):
          #save_image(f'{output_dir}/{img_idx:06d}_transform{d_idx*DSIZE+i:03d}_rem.png', output_images[i])
          mix_video.write(output_images[i])

      del mix_video


if __name__ == '__main__':
  main()
