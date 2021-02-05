import gin
import torch
import torch.nn.functional as F


@gin.configurable
def focal_loss(
    input_tensor: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1,
    gamma: float = 2.0,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Function implements focal loss function.
    Focal loss defined as

    FL(p_t) = -a_t(1-p_t)^g * log(p_t) -- a is alpha and g is gamma

    Reference: https://arxiv.org/pdf/1708.02002.pdf

    Args:
        input_tensor:
        target:
        alpha:
        gamma:
        reduction:
        eps:

    Returns:

    """
    batch_size = input_tensor.size(0)
    out_size = (batch_size,) + input_tensor.size()[2:]
    if target.size()[1:] != input_tensor.size()[2:]:
        raise ValueError(
            "Expected target size {}, got {}".format(out_size, target.size())
        )

    # Compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input_tensor, dim=1) + eps

    # Create the labels one hot tensor, dim 1 as class axis
    target_one_hot = F.one_hot(target, num_classes=input_tensor.shape[1]).view(
        (target.shape[0], -1, *target.shape[1:])
    )

    weight = torch.pow(1.0 - input_soft, gamma)
    focal = -alpha * weight * torch.log(input_soft)
    loss_tmp = torch.sum(
        target_one_hot * focal, dim=1
    )  # corresponds to "indexing" the expected class

    if reduction == "none":
        loss = loss_tmp
    elif reduction == "mean":
        loss = torch.mean(loss_tmp)
    elif reduction == "sum":
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError("Invalid reduction mode: {}".format(reduction))

    return loss
