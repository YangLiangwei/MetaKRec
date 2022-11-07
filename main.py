import os
import argparse
import torch
import logging
import time
from train import main
from utils import seed_everything, logging_setting
import warnings
from utils.mydataset import *

warnings.filterwarnings('ignore')

#if __name__ == '__main__':
seed_everything(seed=1234)

t = time.time()

main()

logging.info('Time used: %ds\n\n\n==========================================================================================\n' % (time.time() - t))
logging.info('finished all')
