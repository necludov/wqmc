from absl import app
from absl import flags

import os

import yaml
import munch

import run
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

from jax import config
# config.update('jax_debug_nans', True)
# config.update('jax_platform_name', 'cpu')
# config.update("jax_enable_x64", True)

FLAGS = flags.FLAGS

flags.DEFINE_string("config", None, "Config path.")
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_string("mode", "train", "train or pretrain.")
flags.mark_flags_as_required(["workdir", "config"])

def launch(argv):
  with open(FLAGS.config, "r") as ymlfile:
    config = yaml.safe_load(ymlfile)
  config = munch.munchify(config)
  if FLAGS.mode == 'pretrain':
    run.pretrain(config, FLAGS.workdir)
  elif FLAGS.mode == 'train':
    run.train(config, FLAGS.workdir)
  else:
    NotImplementedError(f'unrecognized mode: {FLAGS.mode}')


if __name__ == "__main__":
  app.run(launch)
