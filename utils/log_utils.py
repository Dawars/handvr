import torch


def string2color(str):
    hash = 0
    for i in range(len(str)):
        hash = ord(str[i]) + ((hash << 5) - hash)

    red = ((hash >> (0 * 8)) & 0xFF) / 255
    green = ((hash >> (1 * 8)) & 0xFF) / 255
    blue = ((hash >> (2 * 8)) & 0xFF) / 255

    return red, green, blue


def get_log_path(name, params):
    """
    Create a unique directory name based on the model parameters
    :param name: descriptive name of the model (e.g. vae, ae, cnn)
    :param params: dictionary of parameters for model
    :return:
    """
    name = name

    for key in params.keys():
        value = params[key]
        if not isinstance(value, (int, float, bool, str)):
            value = value.__name__
        name += "-{key}_{value}".format(key=key, value=value)

    return name


if __name__ == '__main__':
    params = {
        'dropout': 0.6,
        'act': torch.nn.SELU,
        'batch': 128,
        'l_rate': '1e-5',
    }

    print(get_log_path('test', params))

    print(string2color(get_log_path("color2", params)))
