import unittest
import torch
from src.multi_task_estimator import MultiTaskEstimator


class TestMultiTaskEstimator(unittest.TestCase):
    def test_multi_task_estimator(self):
        num_tasks = 3
        user_id_hash_size = 100
        user_id_embedding_dim = 50
        user_features_size = 10
        item_id_hash_size = 200
        item_id_embedding_dim = 30
        item_features_size = 10
        cross_features_size = 10
        batch_size = 3

        # unused in the baseline MultiTaskEstimator implementation
        user_value_weights = [0.5, 0.3, 0.2]
        assert len(user_value_weights) == num_tasks

        # Instantiate the MultiTaskEstimator
        model: MultiTaskEstimator = MultiTaskEstimator(
            num_tasks, user_id_hash_size, user_id_embedding_dim,
            user_features_size, item_id_hash_size, item_id_embedding_dim,
            item_features_size, cross_features_size,
            user_value_weights
        )

        # Example input data
        user_id = torch.tensor([1, 2, 3])
        user_features = torch.randn(batch_size, user_features_size)
        item_id = torch.tensor([4, 5, 6])
        item_features = torch.randn(batch_size, item_features_size)
        cross_features = torch.randn(batch_size, cross_features_size)
        position = torch.tensor([1, 2, 3], dtype=torch.int32)
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