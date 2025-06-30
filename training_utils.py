import torch
import torch.nn as nn

def tangent_space_criterion(z_dot, basis_vectors1, basis_vectors2=None):  # derivatives_shape (batch, 128*9, 6) # z_dot_shape (batch, 128, 3, 3)
    vectorized_z_dot = z_dot.view(z_dot.shape[0], -1)
    vectorized_basis_vectors1 = basis_vectors1.view(basis_vectors1.shape[0], -1, basis_vectors1.shape[-1])
    basis_vectors_norms1 = torch.norm(vectorized_basis_vectors1, dim=1, keepdim=True)
    normalized_basis_vectors1 = vectorized_basis_vectors1 / basis_vectors_norms1
    z_dot_norm = torch.norm(vectorized_z_dot, dim=1, keepdim=True)
    z_unit = vectorized_z_dot / z_dot_norm
    # z_unit[z_dot_norm.squeeze() == 0, :] = 0

    linear_fit1 = torch.linalg.lstsq(normalized_basis_vectors1, z_unit.unsqueeze(2))  # A.X = B
    residuals1 = torch.bmm(normalized_basis_vectors1, linear_fit1.solution) - z_unit.unsqueeze(2)
    sse1 = torch.sum(residuals1 ** 2, dim=1)
    span_loss1 = torch.mean(sse1)
    span_loss = span_loss1
    pairwise_multiplication = torch.bmm(normalized_basis_vectors1.permute([0, 2, 1]),
                                        normalized_basis_vectors1)
    batch_identity = torch.eye(pairwise_multiplication.shape[1],
                               pairwise_multiplication.shape[2], device=pairwise_multiplication.device).unsqueeze(0)
    orthogonality_loss = torch.mean(
        (batch_identity.expand(pairwise_multiplication.shape[0], -1, -1) - pairwise_multiplication) ** 2)
    smoothness_loss = torch.zeros_like(span_loss)

    if basis_vectors2 is not None:
        vectorized_basis_vectors2 = basis_vectors2.view(basis_vectors2.shape[0], -1, basis_vectors2.shape[-1])
        basis_vectors_norms2 = torch.norm(vectorized_basis_vectors2, dim=1, keepdim=True)
        normalized_basis_vectors2 = basis_vectors2 / basis_vectors_norms2
        basis_vectors_diff = (
                    normalized_basis_vectors1 - normalized_basis_vectors2)  # alt: measure the difference between the spaces that these vectors span
        smoothness_loss = torch.mean(torch.sum(basis_vectors_diff ** 2, dim=2))
        linear_fit2 = torch.linalg.lstsq(vectorized_basis_vectors2, -1 * z_unit.unsqueeze(2))
        residuals2 = torch.bmm(basis_vectors2, linear_fit2.solution) - (-1 * z_unit.unsqueeze(2))
        sse2 = torch.sum(residuals2 ** 2, dim=1)
        span_loss2 = torch.mean(sse2)
        span_loss = (span_loss1 + span_loss2) / 2

    # orthogonality_loss = torch.zeros_like(span_loss)
    norm_loss = torch.zeros_like(span_loss)

    return norm_loss, orthogonality_loss, span_loss, smoothness_loss

def directional_derivative_criterion(x1, x2, x_hat1, x_hat2, z1, z2, directional_derivative=None):
    # x1 -f-> z1 -g-> x_hat1

    x1_vec = x1.view(x1.shape[0], -1)
    x2_vec = x2.view(x2.shape[0], -1)
    v = z2 - z1
    x1_reconstruction_loss = torch.norm(x1_vec - x_hat1, dim=1).mean()
    x2_reconstruction_loss = torch.norm(x2_vec - x_hat2, dim=1).mean()
    directional_derivative_loss = None
    if directional_derivative is not None:
        directional_derivative_loss = torch.norm(directional_derivative - (x2_vec -x1_vec), dim=1).mean()

    return x1_reconstruction_loss, x2_reconstruction_loss, directional_derivative_loss


