import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn import ConvEncoder, ConvDecoder
from .classifier import NonLinClassifier


def build_simplex_dirs(classes, embedding_dim):
    if classes < 2:
        return torch.zeros(classes, embedding_dim)
    n_anomaly_classes = classes - 1
    if embedding_dim < n_anomaly_classes:
        raise ValueError(
            f'embedding_dim={embedding_dim} must be >= classes-1={n_anomaly_classes} '
            'for simplex anomaly directions.'
        )
    if n_anomaly_classes == 1:
        dirs = torch.zeros(classes, embedding_dim)
        dirs[1, 0] = 1.0
        return dirs
    eye = torch.eye(n_anomaly_classes)
    centered = eye - torch.ones_like(eye) / n_anomaly_classes
    centered = F.normalize(centered, p=2, dim=1)
    dirs = torch.zeros(classes, embedding_dim)
    dirs[1:, :n_anomaly_classes] = centered
    return dirs


class simpa(nn.Module):
    is_simpa = True

    def __init__(self, params):
        super().__init__()

        self.name = params.name
        self.classes = params.classes
        self.c_loss_ratio = params.c_loss_ratio
        self.smoothing_alpha = params.smoothing_alpha
        self.smoothing_beta = params.smoothing_beta
        self.ae_mask_ratio = getattr(params, 'ae_mask_ratio', 0.15)
        self.ae_noise_std = getattr(params, 'ae_noise_std', 0.05)

        num_inputs = params.n_features
        seq_len = params.n_time
        num_filters = params.num_filters
        embedding_dim = params.embedding_dim
        kernel_size = params.kernel_size
        dropout = params.dropout
        normalization = params.normalization
        stride = params.stride
        padding = params.padding
        classifier_dim = params.classifier_dim

        self.encoder = ConvEncoder(
            num_inputs,
            num_filters,
            embedding_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dropout=dropout,
            normalization=normalization,
        )
        self.decoder = ConvDecoder(
            embedding_dim,
            num_filters,
            seq_len,
            num_inputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dropout=dropout,
            normalization=normalization,
        )
        self.classifier = NonLinClassifier(
            embedding_dim,
            self.classes,
            d_hidd=classifier_dim,
            dropout=dropout,
            norm=normalization,
        )

        self.sc_loss_ratio = getattr(params, 'sc_loss_ratio', 0.001)
        self.sc_radius_scale = getattr(params, 'sc_radius_scale', 0.25)
        self.eps_center = getattr(params, 'eps_center', 1e-3)
        self.head_init_noise_std = getattr(params, 'head_init_noise_std', 1e-3)

        self.proj1 = nn.Linear(embedding_dim, embedding_dim)
        self.proj2 = nn.Linear(embedding_dim, embedding_dim)
        self._init_projection_heads()

        self.register_buffer('center', torch.zeros(embedding_dim))
        self.register_buffer('simplex_dirs', build_simplex_dirs(self.classes, embedding_dim))
        self.register_buffer('anchor_radius', torch.zeros(self.classes))
        self.register_buffer('center_initialized', torch.tensor(False))

    def _init_projection_heads(self):
        if self.proj1.weight.shape[0] == self.proj1.weight.shape[1]:
            nn.init.eye_(self.proj1.weight)
            nn.init.eye_(self.proj2.weight)
            with torch.no_grad():
                self.proj1.weight.add_(
                    torch.randn_like(self.proj1.weight) * self.head_init_noise_std
                )
                self.proj2.weight.add_(
                    torch.randn_like(self.proj2.weight) * self.head_init_noise_std
                )
        else:
            nn.init.xavier_uniform_(self.proj1.weight)
            nn.init.xavier_uniform_(self.proj2.weight)
        nn.init.zeros_(self.proj1.bias)
        nn.init.zeros_(self.proj2.bias)

    def corrupt_input(self, x):
        if not self.training:
            return x

        corrupted = x
        if self.ae_mask_ratio > 0:
            keep_mask = torch.rand_like(corrupted) >= self.ae_mask_ratio
            corrupted = corrupted * keep_mask.to(corrupted.dtype)
        if self.ae_noise_std > 0:
            corrupted = corrupted + torch.randn_like(corrupted) * self.ae_noise_std
        return corrupted

    def forward_simpa(self, x):
        x_enc = self.encoder(self.corrupt_input(x))
        x_hat = self.decoder(x_enc)
        h = x_enc.reshape(x_enc.size(0), -1)
        z1 = self.proj1(h)
        z2 = self.proj2(h)
        cls_input = (z1 + z2) / 2
        x_out = self.classifier(cls_input)
        return x_hat, x_out, x_enc, z1, z2, h

    def forward(self, x):
        return self.forward_simpa(x)

    def calculate_loss(
        self,
        inputs,
        predicted,
        label,
        pred_label,
        radius,
        z1,
        z2,
    ):
        loss_ae_fn = nn.MSELoss()
        loss_c_fn = nn.CrossEntropyLoss(reduction='none')

        loss_ae = loss_ae_fn(inputs, predicted)

        normal_loc = 0
        label = (
            label * (1 - self.smoothing_alpha - self.smoothing_beta * self.classes + self.smoothing_beta)
            + (1 - label) * self.smoothing_beta
        )
        label[:, normal_loc] += self.smoothing_alpha

        loss_c = torch.mean(loss_c_fn(pred_label, label))
        base_loss = (1 - self.c_loss_ratio) * loss_ae + self.c_loss_ratio * loss_c

        zero = base_loss.new_tensor(0.0)
        stats = {}
        if self.sc_loss_ratio <= 0 or not bool(self.center_initialized.item()):
            return base_loss, loss_ae, loss_c, zero, stats

        label_id = label.argmax(dim=1)
        radius = radius.to(z1.device).float()
        target = self.center.unsqueeze(0) + radius.unsqueeze(1) * self.simplex_dirs[label_id]
        d1 = torch.sum((z1 - target) ** 2, dim=1)
        d2 = torch.sum((z2 - target) ** 2, dim=1)
        var = (d1 - d2) ** 2
        loss_sc = torch.mean(0.5 * torch.exp(-var.clamp(max=50.0)) * (d1 + d2) + 0.5 * var)
        loss = base_loss + self.sc_loss_ratio * loss_sc
        return loss, loss_ae, loss_c, loss_sc, stats

    def initialize_geometry(self, train_dataloader, device):
        was_training = self.training
        self.eval()
        y_windows = getattr(train_dataloader, 'Y_windows', None)
        labels = getattr(train_dataloader, 'label', None)
        if y_windows is None or labels is None:
            raise RuntimeError('Cannot initialize simpa geometry: train loader has no materialized windows.')

        normal_indices = torch.nonzero(labels[:, 0] == 1, as_tuple=False).reshape(-1)
        if normal_indices.numel() == 0:
            raise RuntimeError('Cannot initialize simpa geometry: no normal samples in train loader.')

        z_list = []
        batch_size = getattr(train_dataloader, 'batch_size', 128)
        with torch.no_grad():
            for batch_indices in torch.split(normal_indices, batch_size):
                inputs = y_windows[batch_indices].transpose(2, 1).to(device)
                _, _, _, z1, _, _ = self.forward_simpa(inputs)
                z_list.append(z1)

        z_all = torch.cat(z_list, dim=0)
        center = z_all.mean(dim=0)
        near_zero = torch.logical_and(center.abs() < self.eps_center, center != 0)
        center = center.clone()
        center[near_zero] = self.eps_center * torch.sign(center[near_zero])

        anchor_radius = torch.zeros_like(self.anchor_radius, device=device)
        if anchor_radius.numel() > 1:
            anchor_radius[1:] = float(self.sc_radius_scale) / 2.0

        self.center.copy_(center)
        self.anchor_radius.copy_(anchor_radius)
        self.center_initialized.copy_(torch.tensor(True, device=device))

        if was_training:
            self.train()

        return {
            'normal_count': int(normal_indices.numel()),
            'sc_radius_scale': float(self.sc_radius_scale),
            'anchor_radius': self.anchor_radius.detach().cpu().numpy().tolist(),
        }


def build_model(params):
    return simpa(params)
