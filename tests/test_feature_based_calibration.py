import unittest
import torch
from src.feature_based_calibration import FeatureBasedCalibration, compute_calibration_mse_separated_by_feature


class TesFeatureBasedCalibration(unittest.TestCase):
    def test_compute_calibration_mse_separated_by_feature_2(self):
        B = 3
        T = 2
        preds: torch.Tensor = torch.Tensor([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6]
        ])
        labels: torch.Tensor = torch.Tensor([
            [0, 1],
            [0, 1],
            [1, 0]
        ])
        binary_feature: torch.Tensor = torch.Tensor([
            0.01, 0.8, 0.99
        ])
        result: torch.Tensor = compute_calibration_mse_separated_by_feature(
            preds=preds, labels=labels, binary_feature=binary_feature, num_tasks=T
        )
        self.assertIsInstance(result, torch.Tensor)
        # mean pred for ~ 0 = [0.1, 0.2]
        # mean label for ~ 0 = [0, 1]
        # diff = [0.1, 0.8], cali = 0.01 + 0.64
        # mean pred for ~ 1 = [0.4, 0.5]
        # mean label for ~ 1 = [0.5, 0.5] 
        # diff = [0.1, 0], cali = 0.01 + 0
        # Net cali = 0.01 + 0.64 + 0.01 = 0.66
        assert torch.allclose(result, torch.tensor(0.66))

    def test_compute_calibration_mse_separated_by_feature_1(self):
        B = 3
        T = 1
        preds: torch.Tensor = torch.Tensor([
            [0.1],
            [0.5],
            [0.7]
        ])
        labels: torch.Tensor = torch.Tensor([
            [0],
            [1],
            [1]
        ])
        binary_feature: torch.Tensor = torch.Tensor([
            0.01, 0.8, 0.9
        ])
        result: torch.Tensor = compute_calibration_mse_separated_by_feature(
            preds=preds, labels=labels, binary_feature=binary_feature, num_tasks=T
        )
        self.assertIsInstance(result, torch.Tensor)
        # This is why the value makes sense.
        # 0.17 = 0.16 + 0.01 = 0.4**2 + 0.1**2
        # The mean pred and label for feature ~ 0 = 0.1 and 0. Hence diff = 0.1
        # The mean pred and label for feature ~ 1 = 0.6 and 1. Hence diff = 0.4
        assert torch.allclose(result, torch.tensor(0.17))

    def test_feature_cali(self):
        num_tasks = 3
        user_id_hash_size = 100
        user_id_embedding_dim = 50
        user_features_size = 10
        item_id_hash_size = 200
        item_id_embedding_dim = 30
        item_features_size = 10
        cross_features_size = 10
        batch_size = 8
        num_pred_buckets: int = 4
        cali_loss_wt = 0.1

        # unused in the baseline MultiTaskEstimator implementation
        user_value_weights: list[float] = [0.5, 0.3, 0.2]
        assert len(user_value_weights) == num_tasks

        # Instantiate the MultiTaskEstimator
        model: FeatureBasedCalibration = FeatureBasedCalibration(
            num_tasks, user_id_hash_size, user_id_embedding_dim,
            user_features_size, item_id_hash_size, item_id_embedding_dim,
            item_features_size, cross_features_size,
            user_value_weights,
            training_batch_size=batch_size,
            cali_user_feature_index=2,
            cali_loss_wt=cali_loss_wt,
        )

        # Example input data
        user_id = torch.arange(1, batch_size + 1)
        user_features = torch.randn(batch_size, user_features_size)
        item_id = torch.arange(1, batch_size + 1)
        item_features = torch.randn(batch_size, item_features_size)
        cross_features = torch.randn(batch_size, cross_features_size)
        position = torch.randint(1, 5, size=(batch_size,), dtype=torch.int32)
        labels = torch.randint(2, size=(batch_size, num_tasks))

        # Example train_forward pass
        loss = model.train_forward(
            user_id, user_features,
            item_id, item_features,
            cross_features, position,
            labels
        )
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreaterEqual(loss.item(), 0)

        # Example forward pass
        inference_position = torch.zeros(batch_size, dtype=torch.int32)
        output = model(
            user_id, user_features,
            item_id, item_features,
            cross_features, inference_position
        )
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (batch_size, num_tasks))



if __name__ == '__main__':
    unittest.main()