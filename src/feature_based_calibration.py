"""
We measure miscalibration when the dataset is divided based on the categorical values of the specific feature.
"""

from typing import List

import torch
import torch.nn.functional as F

from src.multi_task_estimator import MultiTaskEstimator


def compute_calibration_mse_separated_by_feature(
    preds: torch.Tensor,
    labels: torch.Tensor, 
    binary_feature: torch.Tensor,
    num_tasks: int
) -> torch.Tensor:
    """
    Args:
        preds: Tensor of shape [B, T]
        labels: Tensor of shape [B, T]
        binary_feature: Tensor of shape [B] with float values (ideally ~0 or ~1)
        num_tasks: int
    Returns:
        mse_lt_05: Tensor of shape [T] (MSE for rows where f < 0.5)
        mse_gte_05: Tensor of shape [T] (MSE for rows where f >= 0.5)
    """
    # Create masks for which indices in the batch are lt or gt 0.5
    mask_lt_05: torch.Tensor = (binary_feature < 0.5).unsqueeze(-1)  # [B, 1]
    mask_gte_05: torch.Tensor = (binary_feature >= 0.5).unsqueeze(-1)  # [B, 1]

    # Apply masks to preds and labels
    preds_lt_05: torch.Tensor = preds[mask_lt_05.expand_as(preds)].view(-1, num_tasks)
    labels_lt_05: torch.Tensor = labels[mask_lt_05.expand_as(labels)].float().view(-1, num_tasks)
    preds_gte_05: torch.Tensor = preds[mask_gte_05.expand_as(preds)].view(-1, num_tasks)
    labels_gte_05: torch.Tensor = labels[mask_gte_05.expand_as(labels)].float().view(-1, num_tasks)

    # Compute means of pred and label for lt and gte partitions
    # This is because the mean of preds_lt_05 is the per-task mean
    # prediction for the case when feature is ~0
    # Similarly label mean is the mean of the observation for this
    # value of the binary feature.
    # Calibration for us means that these should be the same and hence
    # we are trying to compute the difference between these mean values.
    preds_lt_05_mean: torch.Tensor = torch.mean(preds_lt_05, dim=0)
    labels_lt_05_mean: torch.Tensor = torch.mean(labels_lt_05, dim=0)
    preds_gte_05_mean: torch.Tensor = torch.mean(preds_gte_05, dim=0)
    labels_gte_05_mean: torch.Tensor = torch.mean(labels_gte_05, dim=0)

    # Compute MSE for each condition
    return ((preds_lt_05_mean - labels_lt_05_mean) ** 2).sum() + ((preds_gte_05_mean - labels_gte_05_mean) ** 2).sum()

class FeatureBasedCalibration(MultiTaskEstimator):
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
        training_batch_size: int,
        cali_user_feature_index: int,
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
            training_batch_size (B): size of training batch
            cali_user_feature_index: the user feature to use for calibration
            cali_loss_wt: weight for calibration loss.
        """
        super(FeatureBasedCalibration, self).__init__(
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
        self.training_batch_size: int = training_batch_size
        self.cali_user_feature_index: int = cali_user_feature_index
        self.cali_loss_wt: float = cali_loss_wt

    def train_forward(
        self,
        user_id,
        user_features,
        item_id,
        item_features,  # [B, II]
        cross_features,  # [B, IC]
        position,  # [B]
        labels: torch.Tensor,  # [B, T]
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
        binary_feature: torch.Tensor = user_features[:,self.cali_user_feature_index]
        calibration_loss: torch.Tensor = compute_calibration_mse_separated_by_feature(
            preds=torch.sigmoid(ui_logits),
            labels=labels.float(),
            binary_feature=binary_feature,
            num_tasks=self.num_tasks
        )
        return cross_entropy_loss + calibration_loss * self.cali_loss_wt
