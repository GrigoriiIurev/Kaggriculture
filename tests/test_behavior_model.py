import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.kaggriculture.core.action_codec import (
    ARGUMENTS,
    ARGUMENT_TO_ID,
    WORKER_OPERATION_TO_ID,
    WORKER_OPERATIONS,
)
from src.kaggriculture.data.worker_dataset import WorkerFeatureExtractor
from src.kaggriculture.learning.behavior_model import BehaviorCloningPolicy
from src.kaggriculture.learning.train_behavior_cloning import (
    export_model,
    train_models,
)


class BehaviorModelTests(unittest.TestCase):
    def test_training_and_export_support_variable_worker_quantities(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "workers.jsonl.gz"
            with gzip.open(dataset_path, "wt", encoding="utf-8") as output:
                for index in range(30):
                    operation = ("PASS", "PICKUP", "PLACE")[index % 3]
                    argument = "NONE" if operation == "PASS" else (
                        "WHEAT" if index % 2 else "COW"
                    )
                    record = {
                        "split": "train",
                        "feature_indices": [index % 4],
                        "feature_values": [1.0],
                        "workers": [
                            {
                                "context_indices": [],
                                "context_values": [],
                                "is_farmer": True,
                                "target": {
                                    "operation_id": WORKER_OPERATION_TO_ID[operation],
                                    "argument_id": ARGUMENT_TO_ID[argument],
                                    "quantity": 2 if index % 2 else 7,
                                },
                            }
                        ],
                    }
                    output.write(json.dumps(record) + "\n")

            operation_model, argument_model, quantity_model, labels = train_models(
                dataset_path, feature_count=4, epochs=1, batch_size=8
            )
            model_path = Path(directory) / "worker.npz"
            export_model(
                model_path,
                4,
                operation_model,
                argument_model,
                quantity_model,
            )

            self.assertEqual(labels["quantities"][2], 10)
            self.assertEqual(labels["quantities"][7], 10)
            with np.load(model_path, allow_pickle=False) as model:
                self.assertEqual(int(model["version"][0]), 2)
                self.assertIn("quantity_coef", model.files)

    def test_version_two_model_predicts_worker_quantity(self):
        feature_count = WorkerFeatureExtractor(board_size=2).feature_count
        argument_feature_count = feature_count + len(WORKER_OPERATIONS)
        quantity_feature_count = argument_feature_count + len(ARGUMENTS)
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "worker.npz"
            np.savez_compressed(
                model_path,
                version=np.asarray([2], dtype=np.int16),
                feature_count=np.asarray([feature_count], dtype=np.int32),
                operation_classes=np.asarray([0], dtype=np.int16),
                operation_coef=np.zeros((1, feature_count), dtype=np.float32),
                operation_intercept=np.zeros(1, dtype=np.float32),
                argument_classes=np.asarray([0], dtype=np.int16),
                argument_coef=np.zeros(
                    (1, argument_feature_count), dtype=np.float32
                ),
                argument_intercept=np.zeros(1, dtype=np.float32),
                argument_feature_count=np.asarray(
                    [argument_feature_count], dtype=np.int32
                ),
                quantity_coef=np.zeros(quantity_feature_count, dtype=np.float32),
                quantity_intercept=np.asarray([np.log1p(7)], dtype=np.float32),
                quantity_feature_count=np.asarray(
                    [quantity_feature_count], dtype=np.int32
                ),
                quantity_transform=np.asarray(["log1p"]),
            )

            policy = BehaviorCloningPolicy(model_path, board_size=2)

        empty_indices = np.asarray([], dtype=np.int32)
        empty_values = np.asarray([], dtype=np.float32)
        self.assertEqual(
            policy._predict_quantity(
                empty_indices,
                empty_values,
                WORKER_OPERATION_TO_ID["PICKUP"],
                ARGUMENT_TO_ID["WHEAT"],
            ),
            7,
        )
        self.assertEqual(
            policy._predict_quantity(
                empty_indices,
                empty_values,
                WORKER_OPERATION_TO_ID["PASS"],
                ARGUMENT_TO_ID["NONE"],
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
