import unittest

from nimesh.mixins import Named, ListOfNamed


class TestListOfNamed(unittest.TestCase):
    """Test the nimesh.mixin.ListOfNamed class"""

    def test_getitem(self):
        """Test the __getitem__ property"""

        a = Named('a')
        b = Named('b')
        items = ListOfNamed([a, b])

        should_be_a = items['a']
        self.assertEqual(should_be_a, a)
        should_be_a = items[0]
        self.assertEqual(should_be_a, a)

        should_be_b = items['b']
        self.assertEqual(should_be_b, b)
        should_be_b = items[1]
        self.assertEqual(should_be_b, b)

        self.assertIsNone(items['c'])
