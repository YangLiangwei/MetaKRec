import os
import logging
def logging_setting(args):
    if not os.path.exists('./log'):
        os.mkdir('./log')
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S"

    log_name = './log/{}_dim{}_num_layers.{}_lr.{}_weight_decay.{}.log'.format(args.dataset, args.dim, args.num_layers, args.lr, args.l2_weight_decay)
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=[logging.FileHandler(log_name), logging.StreamHandler()])
    return log_name
if __name__ == '__main__':
    print('la')
