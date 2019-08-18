import logging
import json
import argparse
import os
from code.manage import manage
from code.evaluation import crossValidation
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def load_options():
    parser = argparse.ArgumentParser(prog='subjectivity-classification')
    parser.add_argument('choice', choices=['train', 'test', 'classify', 'interactive'])
    parser.add_argument('--input', required=False, help='input .txt file')
    parser.add_argument('--output', required=False, nargs='?',const='output.txt', default='output.txt')
    parser.add_argument('-preprocess', dest='toPreprocess', action='store_true')
    parser.add_argument('-useNB', dest='useNB', action='store_true')
    parser.add_argument('-correlated', dest='toCorrelate', action='store_true')
    args = parser.parse_args()
    if(args.choice == 'classify' and args.input is None):
        parser.error("input file is required")
    return args

def run():
    '''get options, config, logger and run option'''
    file_path = os.path.split(os.path.abspath(__file__))[0]
    options = load_options()
    logging.basicConfig(filename='loggingInfo.log',
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info("Starting logging with options: \n{} \n".format(options))
    if(options.choice=='train'):
        manage.train(options)
    elif(options.choice=='test'):
        print(crossValidation.CV(options, file_path, correlated=options.toCorrelate))
    elif(options.choice=='classify'):
        manage.file_classify(logger, options, file_path, correlated=options.toCorrelate)
    elif(options.choice=='interactive'):
        manage.interactive(options, file_path)


if __name__ == '__main__':
    run()
