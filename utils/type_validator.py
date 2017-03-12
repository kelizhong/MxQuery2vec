import warnings

NONE, MEDIUM, STRONG = 0, 1, 2


def accepts(*types, **kw):
    """Function decorator. Checks decorated function's arguments are
    of the expected types.

    Parameters:
    types -- The expected types of the inputs to the decorated function.
             Must specify type for each parameter.
    kw    -- Optional specification of 'level' level (this is the only valid
             keyword argument, no other should be given).
             level = ( NONE | MEDIUM | STRONG )
             None: not check the type
             MEDIUM: only show the warning msg, not raise an error
             STRONG: raise an error when wrong type

    """
    if not kw:
        # default level: MEDIUM
        level = 1
    else:
        level = kw['level']
    try:
        def decorator(f):
            def newf(*args):
                if level is NONE:
                    return f(*args)
                assert len(args) == len(types)
                argtypes = tuple(map(type, args))
                if argtypes != types:
                    msg = info(f.__name__, types, argtypes, 0)
                    if level is MEDIUM:
                        warnings.warn('TypeWarning: {}'.format(msg))
                    elif level is STRONG:
                        raise TypeError(msg)
                return f(*args)
            newf.__name__ = f.__name__
            return newf
        return decorator
    except KeyError, key:
        raise KeyError("{} is not a valid keyword argument".format(key))
    except TypeError, msg:
        raise TypeError(msg)


def info(fname, expected, actual, flag):
    """Convenience function returns nicely formatted error/warning msg."""
    format = lambda types: ', '.join([str(t).split("'")[1] for t in types])
    expected, actual = format(expected), format(actual)
    msg = "'{}' method ".format( fname )\
          + ("accepts", "returns")[flag] + " ({}), but ".format(expected)\
          + ("was given", "result is")[flag] + " ({})".format(actual)
    return msg
