import redis


def operator_status(func):
    """Get operation status
    """

    def gen_status(*args, **kwargs):
        error, result = None, None
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            error = str(e)

        return {'result': result, 'error': error}

    return gen_status


class RedisHelper(object):
    def __init__(self, host, port=6379, db=0):
        """
        Redis connector pool helper
        :param host: redis host
        :param port: redis port, default 6379
        :param db: redis db, default 0
        """
        if not hasattr(RedisHelper, 'pool'):
            RedisHelper.create_pool(host, port, db)
        self._connection = redis.Redis(connection_pool=RedisHelper.pool)

    @staticmethod
    def create_pool(host, port, db):
        RedisHelper.pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db)

    @operator_status
    def set_data(self, key, value):
        """set data with (key, value)
        """
        return self._connection.set(key, value)

    @operator_status
    def get_data(self, key):
        """get data by key
        """
        return self._connection.get(key)

    @operator_status
    def del_data(self, key):
        """delete cache by key
        """
        return self._connection.delete(key)