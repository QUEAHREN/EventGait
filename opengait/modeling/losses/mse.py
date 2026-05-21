import torch
from .base import BaseLoss


class MSELoss(BaseLoss):
    def __init__(self, loss_term_weight=1.0):
        super(MSELoss, self).__init__(loss_term_weight)

    def forward(self, student_feat, teacher_feat):
        """
            Compute Mean Squared Error between student and teacher features.

            Expected shapes (flexible):
            - student_feat: [N, ...]  (e.g., [n*s, h*w, c])
            - teacher_feat: [N, ...]  (same shape as student_feat)

            Returns:
            - mean_loss: scalar tensor, global mean over all elements
            - info: dict with 'loss' (mean) and 'hard_loss' (max elementwise loss)
        """
        student = student_feat.float()
        teacher = teacher_feat.float()

        # Element-wise squared error
        loss_map = (student - teacher) ** 2

        # Reduce: keep batch as first dim, flatten the rest
        n = loss_map.size(0)
        loss_flat = loss_map.view(n, -1)
        mean_loss = loss_flat.mean()
        hard_loss = loss_flat.max()

        self.info.update({
            'loss': mean_loss.detach().clone(),
            'hard_loss': hard_loss.detach().clone()
        })

        return mean_loss, self.info


if __name__ == "__main__":
    loss_func = MSELoss()
    # Example shapes similar to docstring: [n*s, h*w, c]
    student = torch.randn(2, 128*64, 8)
    teacher = torch.randn(2, 128*64, 8)
    loss, info = loss_func(student, teacher)
    print(loss)