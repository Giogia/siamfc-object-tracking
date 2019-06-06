from collections import namedtuple

from localconfig import config

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

    parameters = {}

    for section in config:
        fields = dict(config.items(section))
        fields = namedtuple(section, fields.keys())(**fields)

        parameters[section] = fields

    return namedtuple('parameters', parameters.keys())(**parameters)


parameters = parse_arguments()
