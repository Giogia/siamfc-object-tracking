import configparser
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

    config = configparser.ConfigParser()

    config.read(PATH)

    parameters = {}

    for section in config.sections():

        fields = dict(config.items(section))
        fields = namedtuple(section, fields.keys())(**fields)

        parameters[section] = fields

    return namedtuple('parameters', parameters.keys())(**parameters)


# Example
'''
parameters = parse_arguments()
print(parameters)
print('\n')
print(parameters.hyperparameters)
print('\n')
print(parameters.hyperparameters.scale_num)
'''