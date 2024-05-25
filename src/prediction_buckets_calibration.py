"""
We try to reduce the second type of calibration error, miscalibration along prediction buckets.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from src.multi_task_estimator import MultiTaskEstimator


class PredictionBucketsCalibration(MultiTaskEstimator):
    """
    Apart from logistic loss used to train each task in this implementation,
    which is y * log(y_hat) + (1 - y) * log(1 - y_hat), we add per-batch losses
    1. We sort the batch based on the predicted logits
    2. We make K bins
    3. For each bin, we compute the mean of y_hat and y
    4. We compute logistic/MSE loss of the mean values.
    """
    def __init__(
        self,
        num_tasks: int,
        user_id_embedding_dim: int,
        user_features_size: int,
        item_id_hash_size: int,
        item_id_embedding_dim: int,
        item_features_size: int,
        cross_features_size: int,
        user_value_weights: List[float],
        num_pred_buckets: int,
        training_batch_size: int,
        cali_loss_wt: float,
    ) -> None:
        """
        params:
            num_tasks (T): The tasks to compute estimates of
            user_id_embedding_dim (DU): internal dimension
            user_features_size (IU): input feature size for users
            item_id_hash_size: the size of the embedding table for items
            item_id_embedding_dim (DI): internal dimension
            item_features_size: (II) input feature size for items
            cross_features_size: (IC) size of cross features
            user_value_weights: T dimensional weights, such that a linear
            combination of point-wise immediate rewards is the best predictor
            of long term user satisfaction.
            num_pred_buckets (PB): the number of buckets to use for prediction score
                calibration
            training_batch_size (B): size of training batch
            cali_loss_wt: weight for calibration loss.
        """
        super(PredictionBucketsCalibration, self).__init__(
            num_tasks=num_tasks,
            user_id_embedding_dim=user_id_embedding_dim,
            user_features_size=user_features_size,
            item_id_hash_size=item_id_hash_size,
            item_id_embedding_dim=item_id_embedding_dim,
            item_features_size=item_features_size,
            cross_features_size=cross_features_size,
            user_value_weights=user_value_weights
        )
        self.num_pred_buckets = num_pred_buckets
        assert(self.num_pred_buckets <= 20, "We are limiting to 20 buckets")
        self.training_batch_size = training_batch_size
        self.cali_loss_wt = cali_loss_wt

        # To make computation faster we assume that training data is provided
        # to us in a fixed batch size. We create a matrix [B, num_pred_buckets]
        # to project the sorted prediction and label values.
        self.scale_proj_mat = torch.zeros(self.training_batch_size, self.num_pred_buckets)
        # Set 1/self.num_pred_buckets elements to sum to 1 in each column
        bucket_size = self.training_batch_size // self.num_pred_buckets
        for col in range(self.num_pred_buckets):
            start_idx = col * bucket_size
            end_idx = start_idx + bucket_size
            self.scale_proj_mat[start_idx:end_idx, col] = 1.0

        # Sum along columns and divide
        column_sums = self.scale_proj_mat.sum(dim=1)
        self.scale_proj_mat = self.scale_proj_mat / column_sums  # [B, PB]
        # Transposing helps because then we can mutliply with [B, T]
        # during loss computation.
        self.scale_proj_mat = self.scale_proj_mat.T  # [PB, B]
        # Register the scale_proj_mat as a buffer
        self.register_buffer('scale_proj_mat', self.scale_proj_mat)


    def train_forward(
        self,
        user_id,
        user_features,
        item_id,
        item_features,  # [B, II]
        cross_features,  # [B, IC]
        position,  # [B]
        labels,  # [B, T]
    ) -> torch.Tensor:
        """Compute the loss during training"""


        # Get task logits using forward method
        ui_logits = super().forward(
            user_id=user_id,
            user_features=user_features,
            item_id=item_id,
            item_features=item_features,
            cross_features=cross_features,
            position=position,
        )
        labels = labels.float()
        # Compute binary cross-entropy loss
        cross_entropy_loss = F.binary_cross_entropy_with_logits(
            input=ui_logits, target=labels, reduction="sum"
        )
        # compute label_mean [PB, T]
        label_mean = torch.matmul(self.scale_proj_mat, labels)
        # compute pred_mean [PB, T]
        preds = torch.sigmoid(ui_logits)
        pred_mean = torch.matmul(self.scale_proj_mat, preds)
        # compute mean squared error
        mse_per_task = ((label_mean - pred_mean) ** 2).mean(dim=0)
        calibration_loss = mse_per_task.mean()
        return cross_entropy_loss + calibration_loss * self.cali_loss_wt
