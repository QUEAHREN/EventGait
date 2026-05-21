import torch
import torch.nn.functional as F

from .base import BaseLoss


class CosineSimilarityLoss(BaseLoss):
	def __init__(self, loss_term_weight=1.0):
		super(CosineSimilarityLoss, self).__init__(loss_term_weight)

	def forward(self, student_feat, teacher_feat):
		"""
			Compute cosine-similarity loss between student and teacher features.

			Expected shapes (flexible):
			- student_feat: [N, ...]  (e.g., [n*s, h*w, c])
			- teacher_feat: [N, ...]  (same shape as student_feat)

			Returns:
			- mean_loss: scalar tensor, global mean of 1 - cosine similarity
			- info: dict with 'loss' (mean) and 'hard_loss' (max loss element)
		"""
		student = student_feat.float()
		teacher = teacher_feat.float()

		n = student.size(0)
		student_flat = student.view(n, -1)
		teacher_flat = teacher.view(n, -1)

		# Normalize to unit vectors before cosine similarity
		student_norm = F.normalize(student_flat, p=2, dim=1, eps=1e-8)
		teacher_norm = F.normalize(teacher_flat, p=2, dim=1, eps=1e-8)

		cos_sim = (student_norm * teacher_norm).sum(dim=1)
		loss_vec = 1.0 - cos_sim

		mean_loss = loss_vec.mean()
		hard_loss = loss_vec.max()

		self.info.update({
			'loss': mean_loss.detach().clone(),
			'hard_loss': hard_loss.detach().clone()
		})

		return mean_loss, self.info


if __name__ == "__main__":
	loss_func = CosineSimilarityLoss()
	student = torch.randn(2, 128 * 64, 8)
	teacher = torch.randn(2, 128 * 64, 8)
	loss, info = loss_func(student, teacher)
	print(loss)
