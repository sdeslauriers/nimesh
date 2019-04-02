from unittest import TestCase

from nimesh import Segmentation


class TestSegmentation(TestCase):
    """Test the nimesh.core.Segmentation class"""

    def test_repr(self):
        """Test the __repr__ method"""

        segmentation = Segmentation('my-seg', [0, 0, 1, 2])
        expected_str = 'Segmentation(name=my-seg, keys=[0 0 1 2])'
        self.assertEqual(expected_str, repr(segmentation))

    def test_str(self):
        """Test the __str__ method"""

        segmentation = Segmentation('my-seg', [0, 0, 1, 2])
        self.assertEqual('my-seg', str(segmentation))
