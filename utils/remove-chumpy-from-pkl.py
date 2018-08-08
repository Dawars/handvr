import cPickle as pickle
import numpy as np
import chumpy as ch

file = '/Users/dawars/projects/handvr/mpi/data/mano/MANO_RIGHT.pkl'


def main():
    with open(file, 'rb') as f:
        model = pickle.load(f)

    for key in model.keys():
        value = undo_chumpy(model[key])
        model[key] = value

    with open(file[:-4] + '_py3.pkl', 'wb') as f:
        pickle.dump(model, f)
    print('done')


# There are chumpy variables so convert them to numpy.
def undo_chumpy(x):
    return x.r if isinstance(x, ch.Ch) else x


if __name__ == '__main__':
    main()
