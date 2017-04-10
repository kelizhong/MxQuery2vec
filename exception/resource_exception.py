"""exception about resource.e.g. fail to load vocabulary, embedding resource"""


class ResourceNotFoundException(Exception):
    def __init__(self, err="Resource not found"):
        Exception.__init__(self, err)
