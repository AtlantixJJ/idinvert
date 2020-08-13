# python 3.6
"""Inverts given images to latent codes with In-Domain GAN Inversion.

Basically, for a particular image (real or synthesized), this script first
employs the domain-guided encoder to produce a initial point in the latent
space and then performs domain-regularized optimization to refine the latent
code.
"""

import os
import threading
import argparse
import pickle
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from dnnlib import tflib

from perceptual_model import PerceptualModel
from utils.logger import setup_logger
from utils.visualizer import adjust_pixel_range
from utils.visualizer import VideoWriter
from utils.visualizer import save_image, load_image, resize_image


def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('model_path', type=str,
                      help='Path to the pre-trained model.')
  parser.add_argument('-o', '--output_dir', type=str, default='',
                      help='Directory to save the results. If not specified, '
                           '`./results/inversion/${IMAGE_LIST}` '
                           'will be used by default.')
  parser.add_argument('--num', type=int, default=16,
                      help='Generate image number')
  parser.add_argument('--gpu_id', type=str, default='0',
                      help='Which GPU(s) to use. (default: `0`)')
  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
  assert os.path.exists(args.image_list)
  output_dir = args.output_dir or f'results/inversion/{image_list_name}'
  logger = setup_logger(output_dir, 'inversion.log', 'inversion_logger')

  logger.info(f'Loading model.')
  tflib.init_tf({'rnd.np_random_seed': 1000})
  with open(args.model_path, 'rb') as f:
    E, _, _, Gs = pickle.load(f)

  # Get input size.
  image_size = E.input_shape[2]
  assert image_size == E.input_shape[3]

  # Build graph.
  logger.info(f'Building graph.')
  sess = tf.get_default_session()
  input_shape = E.input_shape
  input_shape[0] = args.batch_size
  mask_shape = input_shape[:]
  mask_shape[1] = 1
  x = tf.placeholder(tf.float32, shape=input_shape, name='real_image')
  mask = tf.placeholder(tf.float32, shape=mask_shape, name='mask')
  x_255 = (tf.transpose(x, [0, 2, 3, 1]) + 1) / 2 * 255
  latent_shape = Gs.components.synthesis.input_shape
  latent_shape[0] = args.batch_size
  wp = tf.get_variable(shape=latent_shape, name='latent_code')
  x_rec = Gs.components.synthesis.get_output_for(wp, randomize_noise=False)
  x_rec_255 = (tf.transpose(x_rec, [0, 2, 3, 1]) + 1) / 2 * 255
  if args.random_init:
    logger.info(f'  Use random initialization for optimization.')
    wp_rnd = tf.random.normal(shape=latent_shape, name='latent_code_init')
    setter = tf.assign(wp, wp_rnd)
  else:
    logger.info(f'  Use encoder output as the initialization for optimization.')
    w_enc = E.get_output_for(x, phase=False)
    wp_enc = tf.reshape(w_enc, latent_shape)
    setter = tf.assign(wp, wp_enc)



if __name__ == '__main__':
  main()
