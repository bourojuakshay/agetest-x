import unittest
from age_service import AgeService

class TestAgeService(unittest.TestCase):
    def test_age_group_mapping(self):
        service = AgeService()
        
        # Test cases: (age, expected_group)
        test_cases = [
            (5, "Kid"),
            (12, "Kid"),
            (12.9, "Kid"),
            (13, "Teen"),
            (15, "Teen"),
            (17.9, "Teen"),
            (18, "Young Adult"),
            (20, "Young Adult"),
            (24.9, "Young Adult"),
            (25, "Adult"),
            (30, "Adult"),
            (49.9, "Adult"),
            (50, "Senior"),
            (60, "Senior")
        ]
        
        for age, expected in test_cases:
            with self.subTest(age=age):
                result = service._get_age_group(age)
                self.assertEqual(result, expected, f"Failed for age {age}")

if __name__ == "__main__":
    unittest.main()
