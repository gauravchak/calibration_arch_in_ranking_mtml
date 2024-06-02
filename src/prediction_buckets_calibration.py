"""
We try to reduce the second type of calibration error, miscalibration along prediction buckets.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        user_id_hash_size: int,
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
            user_id_hash_size: the size of the embedding table for users
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
            user_id_hash_size=user_id_hash_size,
            user_id_embedding_dim=user_id_embedding_dim,
            user_features_size=user_features_size,
            item_id_hash_size=item_id_hash_size,
            item_id_embedding_dim=item_id_embedding_dim,
            item_features_size=item_features_size,
            cross_features_size=cross_features_size,
            user_value_weights=user_value_weights
        )
        self.num_pred_buckets: int = num_pred_buckets
        assert (self.num_pred_buckets <= 20), "We are limiting to 20 buckets"
        self.training_batch_size: int = training_batch_size
        self.cali_loss_wt: float = cali_loss_wt

        # To make computation faster we assume that training data is provided
        # to us in a fixed batch size. We create a matrix [B, num_pred_buckets]
        # to project the sorted prediction and label values.
        self.scale_proj_mat: torch.Tensor = torch.zeros(self.training_batch_size, self.num_pred_buckets)
        # Set 1/self.num_pred_buckets elements to 1 in each column
        bucket_size: int = self.training_batch_size // self.num_pred_buckets
        for col in range(self.num_pred_buckets):
            start_idx: int = col * bucket_size
            end_idx: int = start_idx + bucket_size
            self.scale_proj_mat[start_idx:end_idx, col] = 1.0

        # Since we want to compute a mean per bin, these non zero values in
        # each bin should not actually be 1 but 1/k where k is the number of
        # non-zero values in that bin (i.e. prediction bucket)
        # Summing on dim=0 computes the sum for each bin.
        bucket_sums: torch.Tensor = self.scale_proj_mat.sum(dim=0)  # [PB]
        # Dividing by bucket_sums ensures that sum for each bucket is 1.
        # Hence matmul with self.scale_proj_mat will in effect compute
        # the mean for the bucket.
        self.scale_proj_mat = self.scale_proj_mat / bucket_sums  # [B, PB]
        # Transposing helps because then we can mutliply with [B, T]
        # during loss computation.
        self.scale_proj_mat = self.scale_proj_mat.T  # [PB, B]
        # # Register the scale_proj_mat as a buffer
        # self.register_buffer('scale_proj_mat', self.scale_proj_mat)


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
        ui_logits: torch.Tensor = super().forward(
            user_id=user_id,
            user_features=user_features,
            item_id=item_id,
            item_features=item_features,
            cross_features=cross_features,
            position=position,
        )

        # Compute binary cross-entropy loss
        cross_entropy_loss: torch.Tensor = F.binary_cross_entropy_with_logits(
            input=ui_logits, target=labels.float(), reduction="sum"
        )

        # Compute ECE loss
        # These steps have been verified here: https://colab.research.google.com/drive/1EkubNvQ3X_fFLOSb6KDbUGCogk02Ae8b#scrollTo=JNzZTr5BoOkg
        preds: torch.Tensor = torch.sigmoid(ui_logits)
        # Assuming preds and labels are of shape [B, T]
        # Sort preds to get indices
        sorted_indices: torch.Tensor = torch.argsort(preds, dim=0)
        sorted_preds: torch.Tensor = torch.gather(input=preds, dim=0, index=sorted_indices)
        sorted_labels: torch.Tensor = torch.gather(input=labels.float(), dim=0, index=sorted_indices)
        # Compute the mean prediction in each bin
        pred_mean_per_bin: torch.Tensor = torch.matmul(self.scale_proj_mat, sorted_preds)  # [PB, T]
        # compute label_mean in the bucket
        label_mean_per_bin: torch.Tensor = torch.matmul(self.scale_proj_mat, sorted_labels)  # [PB, T]
        # compute mean squared error between mean label and prediction
        # in the bucket. Forst compute per task. This will allow us to later reuse
        # any task specific weights set by the user for cross_entropy_loss.
        mse_per_task: torch.Tensor = ((pred_mean_per_bin - label_mean_per_bin)**2).mean(dim=0)
        calibration_loss: torch.Tensor = mse_per_task.mean()
        return cross_entropy_loss + calibration_loss * self.cali_loss_wt
