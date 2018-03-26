class Named(object):

    def __init__(self, name):
        """Mixin for objects with a name."""
        self._name = name

    @property
    def name(self):
        """Returns the name of the object."""
        return self._name
