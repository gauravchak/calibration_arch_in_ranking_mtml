"""
We are trying to solve the same feature based miscalibration from 
feature_based_calibration.py . However here we are approaching it slightly
differently. We will bring the feature all the way to the last layer and learn
a logit bias for each of the T tasks. This can help to memorize and remove any
uncaptured bias. For instance, if for your feature value = 1, for your task T1,
you see that $\frac{avg(PredictedProbability(T1))}{avg(UserActionLabel(T1))}$
is 1.4, then the learned bias could capture ~ ln(1.4) ~ 0.336 . 
Hence after this addition, 
$\frac{avg(PredictedProbability(T1))}{avg(UserActionLabel(T1))}$ = 1.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.multi_task_estimator import MultiTaskEstimator


class FeatureBiasCapture(MultiTaskEstimator):
    """
    Brings the feature to the last layer.
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
        training_batch_size: int,
        cali_user_feature_index: int
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
            training_batch_size (B): size of training batch
            cali_user_feature_index: the user feature to use for calibration
        """
        super(FeatureBiasCapture, self).__init__(
            num_tasks=num_tasks,
            user_id_embedding_dim=user_id_embedding_dim,
            user_features_size=user_features_size,
            item_id_hash_size=item_id_hash_size,
            item_id_embedding_dim=item_id_embedding_dim,
            item_features_size=item_features_size,
            cross_features_size=cross_features_size,
            user_value_weights=user_value_weights
        )
        self.training_batch_size: int = training_batch_size
        self.cali_user_feature_index: int = cali_user_feature_index
        self.map_feature_bias: nn.Module = nn.Linear(
            in_features=1, out_features=self.num_tasks, bias=True
        )

    def forward(
        self,
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
        item_id: torch.Tensor,  # [B]
        item_features: torch.Tensor,  # [B, II]
        cross_features: torch.Tensor,  # [B, IC]
        position: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        combined_features: torch.Tensor = self.process_features(
            user_id=user_id,
            user_features=user_features,
            item_id=item_id,
            item_features=item_features,
            cross_features=cross_features,
            position=position,
        )
        # Compute per-task scores/logits without worrying about feature
        # based calibration
        ui_logits: torch.Tensor = self.task_arch(combined_features)  # [B, T]
        # For each task it learns logit_delta = w * feature + b
        # Then returns ui_logit + logit_delta
        calibration_bias: torch.Tensor = self.map_feature_bias(user_features)  # [B, T]
        return ui_logits + calibration_bias
