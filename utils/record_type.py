from collections import namedtuple


class RecordType(object):
    def __init__(self, typename, strict_fields, verbose=False):
        """ Store all of the field and type data as class methods so they aren't regenerated
            everytime a new named tuple is required"""
        self.typename = typename

        # Store field type and default information in varous formats for easy access by methods ###
        self.strict_fields = strict_fields
        self._strict_names = [v[0] for v in strict_fields]
        self._strict_types = [type(v[1]) for v in strict_fields]
        self.strict_defaults = [v[1] for v in strict_fields]

        vars(self)[typename] = namedtuple(typename, self._strict_names,
                                          verbose=verbose)  # Creates a namedtuple class from factory function

        self.record = self._dict_make(**dict(strict_fields))

    def _typecheck(self, arg, fieldtype, warning=False):
        """Takes in an argument and a field type and trys to recast if necessary, then returns recast argument"""
        if not isinstance(arg, fieldtype):
            try:
                oldarg = arg  # Keep for error printout
                arg = fieldtype(arg)  # Attempt recast
            except (ValueError, TypeError):  # Recast failed
                raise TypeError("Argument: %s to %s" % (arg, fieldtype))
            else:
                if warning:
                    print ("Recasting %s to %s as %s" % (oldarg, fieldtype, arg))
        return arg

    def _dict_make(self, **kwargs):
        """ User can pass a dictionary of attributes in and they will be typechecked/recast.  Similiar to passing
        dictionary directly to namedtuple using **d notation"""
        warning = kwargs.pop('warning', False)

        for name, default in self.strict_fields:
            try:
                value = kwargs[name]
            except KeyError:
                kwargs[name] = default  # Throw the default value in if missing
            else:
                value = self._typecheck(value, type(default), warning)  # Typecheck if found
                kwargs[name] = value

        return vars(self)[self.typename](**kwargs)

    def __call__(self, **kwargs):
        self.record = self._dict_make(**kwargs)

        return self

    def __getattr__(self, name):
        return getattr(self.record, name)

    @property
    def dict(self):
        """Return a new dict which maps field types to their values."""
        return self.record._asdict()

    def iteritems(self):
        """record.iteritems -> an iterator over the (key, value) items in record"""
        for k, v in self.dict.iteritems():
            yield (k, v)
