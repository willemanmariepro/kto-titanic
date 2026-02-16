import unittest
from typing import List

"""
determine the threshold for the count
"""
length_threshold = 7


def count_long_names(names: List[str]) -> int:
    """Returns the count of names longer than the defined threshold."""
    long_names = [name for name in names if len(name) > length_threshold]
    
    return len(long_names)


"""
unit test: check the attended result --> return ok 
"""
class TestNameCounter(unittest.TestCase):
    def test_count_long_names(self):
        # sample of test
        sample_names = ["Guillaume", "Gilles", "Juliette", "Antoine", "François", "Cassandre"]
        
        # Act & Assert
        self.assertEqual(count_long_names(sample_names), 4)

if __name__ == '__main__':
    unittest.main()