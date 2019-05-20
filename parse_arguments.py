import configparser
import os
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

        values = dict(config.items(section))
        values = namedtuple(section, values.keys())(**values)

        parameters[section] = values

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