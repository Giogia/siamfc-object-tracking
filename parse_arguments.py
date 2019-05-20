import json
import os
from collections import namedtuple

PATH = 'parameters'


def parse_arguments():

    """

    USAGE
    parameters = parse_arguments()
    value  = parameters.filename.field

    EXAMPLE
    parameters = parse_arguments()
    scale_num = parameters.hyperparams.scale_num

    """

    parameters = {}
    for file in os.listdir(PATH):

        if file.endswith('.json'):
            with open(os.path.join(PATH, file)) as json_file:

                file_name = os.path.splitext(file)[0]
                values = json.load(json_file)
                values = namedtuple(file_name, values.keys())(**values)

                parameters[file_name] = values

    return namedtuple('parameters', parameters.keys())(**parameters)


# Example
# parameters = parse_arguments()
# print(parameters)
# print(parameters.hyperparams)
# print(parameters.hyperparams.scale_num)
