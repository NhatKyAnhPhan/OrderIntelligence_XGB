import unittest

import numpy as np

from app.predictor import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_feature_vector,
)


class BuildFeatureVectorTests(unittest.TestCase):
    def setUp(self):
        self.numeric_values = {name: 1 for name in NUMERIC_FEATURES}
        self.categories = {
            name: ["alpha", "beta"] for name in CATEGORICAL_FEATURES
        }
        self.categorical_values = {
            name: "beta" for name in CATEGORICAL_FEATURES
        }

    def test_preserves_training_feature_order_and_ordinal_codes(self):
        features = build_feature_vector(
            self.numeric_values,
            self.categorical_values,
            self.categories,
        )

        self.assertEqual(features.shape, (28,))
        self.assertEqual(features.dtype, np.float32)
        np.testing.assert_array_equal(features[:16], np.ones(16, dtype=np.float32))
        np.testing.assert_array_equal(features[16:], np.ones(12, dtype=np.float32))

    def test_rejects_unknown_category_instead_of_using_wrong_code(self):
        self.categorical_values["gender"] = "unknown"

        with self.assertRaisesRegex(ValueError, "không có trong danh mục"):
            build_feature_vector(
                self.numeric_values,
                self.categorical_values,
                self.categories,
            )

    def test_rejects_missing_features(self):
        del self.numeric_values["age"]

        with self.assertRaisesRegex(ValueError, "Thiếu hoặc thừa trường số"):
            build_feature_vector(
                self.numeric_values,
                self.categorical_values,
                self.categories,
            )


if __name__ == "__main__":
    unittest.main()
