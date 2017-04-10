"""exception about data shape"""


class DataShapeNotMatchException(Exception):
    def __init__(self, err="data shape not match"):
        Exception.__init__(self, err)
