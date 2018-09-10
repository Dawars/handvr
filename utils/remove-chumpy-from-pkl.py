import argparse

import cPickle as pickle
import numpy as np
import chumpy as ch


def main(args):
    with open(args.file, 'rb') as f:
        model = pickle.load(f)

    for key in model.keys():
        value = undo_chumpy(model[key])
        model[key] = value

    with open(args.file[:-4] + '_py3.pkl', 'wb') as f:
        pickle.dump(model, f)


# There are chumpy variables so convert them to numpy.
def undo_chumpy(x):
    return x.r if isinstance(x, ch.Ch) else x


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Remove chumpy dependency from pickle files to be read in Python 3")
    parser.add_argument("--file", default='./MANO_RIGHT.pkl')

    main(parser.parse_args())
