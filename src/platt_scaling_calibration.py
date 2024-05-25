from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from src.multi_task_estimator import MultiTaskEstimator


class PlattScalingCalibration(MultiTaskEstimator):
    """
    In this implementation after we compute the logits from MultiTaskEstimator
    we pass them through PlattScaling. We employ some optimizations to ensure
    that original model trains fine.
    Most of the time we train using only the original implementation and only
    K=5% of the time we update the PlattScaling parameters.
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
        scaling_frac: float,
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
            scaling_frac: the fraction of times we should be training 
                PlattScaling
        """
        super(PlattScalingCalibration, self).__init__(
            num_tasks=num_tasks,
            user_id_embedding_dim=user_id_embedding_dim,
            user_features_size=user_features_size,
            item_id_hash_size=item_id_hash_size,
            item_id_embedding_dim=item_id_embedding_dim,
            item_features_size=item_features_size,
            cross_features_size=cross_features_size,
            user_value_weights=user_value_weights
        )
        self.scaling_frac = scaling_frac
        self.weights = nn.Parameter(torch.zeros(num_tasks))  # set device=
        self.bias = nn.Parameter(torch.zeros(num_tasks))  # set device=

    def forward(
        self,
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
        item_id: torch.Tensor,  # [B]
        item_features: torch.Tensor,  # [B, II]
        cross_features: torch.Tensor,  # [B, IC]
        position: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        ui_logits = super().forward(
            user_id, user_features, item_id, item_features, 
            cross_features, position
        )
        calibrated_logits = self.weights * ui_logits + self.bias
    
    def train_forward(
        self,
        user_id,
        user_features,
        item_id,
        item_features,  # [B, II]
        cross_features,  # [B, IC]
        position,  # [B]
        labels,
    ) -> torch.Tensor:
        """Compute the loss during training"""
        # Get task logits using superclass/uncalibrated method
        ui_logits = super().forward(
            user_id=user_id,
            user_features=user_features,
            item_id=item_id,
            item_features=item_features,
            cross_features=cross_features,
            position=position,
        )
        # Some percentage of the time we should scale to
        # learn the scaling parameters.
        if random.random() < self.scaling_frac:
            ui_logits = self.weights * ui_logits.detach() + self.bias
        # Compute binary cross-entropy loss
        cross_entropy_loss = F.binary_cross_entropy_with_logits(
            input=ui_logits, target=labels.float(), reduction="sum"
        )

        return cross_entropy_loss
