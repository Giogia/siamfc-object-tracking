from localconfig import config
from collections import namedtuple

PATH = 'parameters.ini'


def parse_arguments():
    """

    USAGE
    parameters = parse_arguments()
    value  = parameters.section.field

    EXAMPLE
    parameters = parse_arguments()
    scale_num = parameters.hyperparameters.scale_num

    """

    config.read(PATH)

    params = {}

    for section in config:
        fields = dict(config.items(section))
        fields = namedtuple(section, fields.keys())(**fields)

        params[section] = fields

    return namedtuple('parameters', params.keys())(**params)


parameters = parse_arguments()
