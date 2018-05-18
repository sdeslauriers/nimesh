from typing import Union


class Named(object):

    def __init__(self, name):
        """Mixin for objects with a name."""
        self._name = name

    @property
    def name(self):
        """Returns the name of the object."""
        return self._name


class ListOfNamed(list):
    """Represents a list of items that have names."""

    def __getitem__(self, item) -> Union[Named, None]:
        """Allows getting items by their name"""

        if isinstance(item, str):
            return next((o for o in self if o.name == item), None)
        else:
            return super().__getitem__(item)

    def copy(self) -> 'ListOfNamed':
        """Copies a list of named object."""
        return ListOfNamed(self)
