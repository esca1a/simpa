import argparse
import json
import numpy as np
import os
import time

import torch
import torch.nn as nn


from models.meta import build_model
import utils
import datautils
from loaders.load import list_nab_entities, list_yahoo_entities
from metrics import (
    align_window_scores_to_point_labels,
    compute_metrics,
)

EARLY_STOPPING = False



def model_parameters(args):
    params_model = utils.AttrDict(
        name='simpa',
        # Model params
        n_features = args.n_features,
        n_time = args.window_size,
        num_filters = [128, 128, 256, 256],
        embedding_dim = args.embedding_dim,
        kernel_size = 4,
        dropout = 0.2,
        normalization = 'batch',
        stride = 2,
        padding = 2,

        anomaly_types = args.anomaly_types,
        classes = len(args.anomaly_types),
        classifier_dim = 32,
        c_loss_ratio = args.c_loss_ratio,

        smoothing_alpha = 0.1,
        smoothing_beta = 0.01,
        ae_mask_ratio = args.ae_mask_ratio,
        ae_noise_std = args.ae_noise_std,
        sc_loss_ratio = args.sc_loss_ratio,
        sc_radius_scale = args.sc_radius_scale,
        eps_center = args.eps_center,
        head_init_noise_std = args.head_init_noise_std,
    )
    return params_model


class MulticlassAnomalyTrainer:
    def __init__(
        self,
        model_dir = "./training",
        params = None,
        device = 'cpu',
    ):

        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.params = params
        self.epoch = params.epoch
        self.device = device

        self.use_amp = torch.cuda.is_available() and str(self.device).startswith("cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.model = build_model(self.params).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.params.lr)


    def _amp_context(self):
        return torch.cuda.amp.autocast(enabled=self.use_amp)


    def train(self, train_dataloader, val_dataloader=None):
        stop_counter = 0
        best_val_loss = np.inf
        best_state_dict = None
        if not bool(self.model.center_initialized.item()):
            self.model.initialize_geometry(train_dataloader, self.device)

        for epoch in range(self.epoch):
            self.model.train()
            starttime = time.time()
            cum_loss, step_count = 0, 0
            cum_loss_ae, cum_loss_c = 0, 0
            cum_loss_geo = 0
            for step, batch in enumerate(train_dataloader):
                self.optimizer.zero_grad()
                inputs = batch['Y'] #(batch, n_features, window)
                inputs = inputs.transpose(2,1).to(self.device, non_blocking=True) #(batch, window, n_features)
                inputs_normal = batch['Z']
                inputs_normal = inputs_normal.transpose(2,1).to(self.device, non_blocking=True) #(batch, window, n_features)
                label = batch['label'].to(self.device, non_blocking=True)
                if inputs.shape[0]==1: #BatchNorm of Classifier doesn't work if batchsize=1
                    continue

                with self._amp_context():
                    radius = batch['radius'].to(self.device, non_blocking=True)
                    predicted, pred_label, pred_enc, z1, z2, _ = self.model.forward_simpa(inputs)
                    loss, loss_ae, loss_c, loss_geo, geo_stats = self.model.calculate_loss(
                        inputs_normal,
                        predicted,
                        label,
                        pred_label,
                        radius,
                        z1,
                        z2,
                    )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm or 1e9)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if torch.isnan(loss).any():
                    print(f'Detected NaN loss at epoch {epoch}')
                else:
                    cum_loss += loss.item()
                    cum_loss_ae += loss_ae.item()
                    cum_loss_c += loss_c.item()
                    cum_loss_geo += loss_geo.item()
                    step_count += 1
            epoch_loss = cum_loss/step_count
            epoch_loss_ae = cum_loss_ae/step_count
            epoch_loss_c = cum_loss_c/step_count
            epoch_loss_geo = cum_loss_geo/step_count
            epoch_t = time.time() - starttime
            print(
                'Epoch:', epoch,
                '     loss: ', str(epoch_loss)[0:6],
                '     loss_ae: ', str(epoch_loss_ae)[0:6],
                '     loss_c: ', str(epoch_loss_c)[0:6],
                '     loss_geo: ', str(epoch_loss_geo)[0:6],
                '     time: ', str(epoch_t)[0:4], 'sec',
            )

            # early stop
            if val_dataloader:
                val_loss, val_loss_ae, val_loss_c, val_loss_geo = self.validation(val_dataloader, epoch)
                if torch.isnan(val_loss).any():
                    stop_counter += 10
                elif val_loss < best_val_loss:
                    stop_counter = 0
                    best_val_loss = val_loss
                    print("best validation loss is updated", '     loss: ', str(best_val_loss.item())[:6], '     loss_ae: ', str(val_loss_ae.item())[0:6], '     loss_c: ', str(val_loss_c.item())[0:6], '     loss_geo: ', str(val_loss_geo.item())[0:6])
                    best_state_dict = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                    }
                else:
                    stop_counter += 1

            if EARLY_STOPPING and val_dataloader and stop_counter > 9:
                break
        if best_state_dict is None:
            print('No validation checkpoint was saved; saving current model for test.')
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in self.model.state_dict().items()
            }
        torch.save(best_state_dict, f'{self.model_dir}/bestmodel.pkl')
        #################################################################################

    def validation(self, val_dataloader, epoch):
        loss = torch.tensor([0.0], requires_grad=False, device=self.device)
        loss_AE = torch.tensor([0.0], requires_grad=False, device=self.device)
        loss_C = torch.tensor([0.0], requires_grad=False, device=self.device)
        loss_G = torch.tensor([0.0], requires_grad=False, device=self.device)

        self.model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                inputs = batch['Y'] #(batch, n_features, window)
                inputs = inputs.transpose(2,1).to(self.device, non_blocking=True) #(batch, window, n_features)
                inputs_normal = batch['Z']
                inputs_normal = inputs_normal.transpose(2,1).to(self.device, non_blocking=True) #(batch, window, n_features)
                label = batch['label'].to(self.device, non_blocking=True)

                with self._amp_context():
                    radius = batch['radius'].to(self.device, non_blocking=True)
                    predicted, pred_label, pred_enc, z1, z2, _ = self.model.forward_simpa(inputs)
                    loss_aec, loss_ae, loss_c, loss_geo, _ = self.model.calculate_loss(
                        inputs_normal,
                        predicted,
                        label,
                        pred_label,
                        radius,
                        z1,
                        z2,
                    )
                loss += loss_aec
                loss_AE += loss_ae
                loss_C += loss_c
                loss_G += loss_geo
            return loss, loss_AE, loss_C, loss_G


def test(test_dataloader, model_dir, params, device):
    model = build_model(params).to(device)
    model.load_state_dict(torch.load(f'{model_dir}/bestmodel.pkl'))
    use_amp = torch.cuda.is_available() and str(device).startswith("cuda")

    inputs_list = []
    prediction_list = []
    pred_label_list = []
    z1_list = []
    z2_list = []

    model.eval()
    with torch.inference_mode():
        for step, batch in enumerate(test_dataloader):
            inputs = batch['Y'] #(batch, n_features, window)
            inputs = inputs.transpose(2,1).to(device, non_blocking=True) #(batch, window, n_features)

            with torch.cuda.amp.autocast(enabled=use_amp):
                predicted, pred_label, pred_enc, z1, z2, _ = model.forward_simpa(inputs)
                z1_list.append(z1)
                z2_list.append(z2)
            pred_label_list.append(pred_label)
            inputs_list.append(inputs)
            prediction_list.append(predicted)

        inputs_list = torch.cat(inputs_list, dim=0)
        inputs_list = inputs_list.to('cpu').detach().numpy().copy()
        prediction_list = torch.cat(prediction_list, dim=0)
        prediction_list = prediction_list.to('cpu').detach().numpy().copy()

        pred_label_list = torch.cat(pred_label_list, dim=0)
        pred_label_list = pred_label_list.to('cpu').detach().numpy().copy()
        z1_list = torch.cat(z1_list, dim=0).to('cpu').detach().numpy().copy()
        z2_list = torch.cat(z2_list, dim=0).to('cpu').detach().numpy().copy()
        geometry_state = {
            'geometry_type': 'simpa',
            'center': model.center.to('cpu').detach().numpy().copy(),
            'simplex_dirs': model.simplex_dirs.to('cpu').detach().numpy().copy(),
            'anchor_radius': model.anchor_radius.to('cpu').detach().numpy().copy(),
            'sc_radius_scale': float(model.sc_radius_scale),
        }

        return inputs_list, prediction_list, pred_label_list, z1_list, z2_list, geometry_state


def convolve_minmax_score(score, w=50, minmax=True):
    # Create the convolution kernel and reshape it for broadcasting
    b = np.ones((w, 1)) / w  # Shape it as (w, 1) to convolve along the time axis

    # Apply convolution across the time dimension for all features simultaneously
    score = np.apply_along_axis(lambda m: np.convolve(m, b[:, 0], mode='same'), axis=0, arr=score)

    # Min-max normalization (if specified)
    if minmax:
        min_vals = score.min(axis=0, keepdims=True)
        max_vals = score.max(axis=0, keepdims=True)
        score = (score - min_vals) / (max_vals - min_vals + 1e-8)  # Avoid division by zero
    return score

def mse(input, pred, mean=True):
    fn = nn.MSELoss(reduction='none')
    mse_score = np.array(fn(torch.Tensor(input), torch.Tensor(pred)))
    if mse_score.ndim==1: mse_score = np.expand_dims(mse_score, axis=1)
    if mean:
        return np.mean(np.array(mse_score), axis=1)
    else:
        return np.array(mse_score)

def label_score_selected_feature(label, axis=[0]):
    label_copy = np.copy(label)
    label_copy[:,axis] = 0
    label_copy = np.sum(label_copy,axis=1)
    return label_copy

def compute_base_anomaly_score(input, pred, pred_label, threshold=0.05):
    B,W,D = input.shape
    input = input.reshape(B, -1)
    pred = pred.reshape(B, -1)
    mse_score = mse(input, pred)
    mse_score = convolve_minmax_score(mse_score, w=int(W/2))

    mean_label = np.mean(pred_label, axis=0)
    indices = np.where(mean_label > threshold)[0]
    if 0 not in indices: indices = np.insert(indices, 0, 0)
    ce_score = label_score_selected_feature(pred_label, axis=indices)
    ce_score = convolve_minmax_score(ce_score, w=int(W/2))

    anomaly_score = (mse_score + ce_score)/2
    return anomaly_score


def smooth_score(score, w=50):
    score = np.asarray(score)
    if score.ndim == 1:
        score = score.reshape(-1, 1)
    w = max(int(w), 1)
    kernel = np.ones((w,)) / w
    smoothed = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=score)
    return smoothed.reshape(-1) if smoothed.shape[1] == 1 else smoothed


def fit_minmax(score):
    score = np.asarray(score, dtype=float)
    return {
        'min': float(np.nanmin(score)),
        'max': float(np.nanmax(score)),
    }


def apply_minmax(score, stats, clip=False):
    score = np.asarray(score, dtype=float)
    normalized = (score - stats['min']) / (stats['max'] - stats['min'] + 1e-8)
    if clip:
        normalized = np.clip(normalized, 0.0, 1.0)
    return normalized


def simpa_geometry_score(
    z1,
    z2,
    pred_label,
    geometry_state,
    gpnf_threshold=0.05,
):
    if z1 is None or z2 is None or geometry_state is None:
        return None, {}

    z1 = np.asarray(z1, dtype=float)
    z2 = np.asarray(z2, dtype=float)
    pred_label = np.asarray(pred_label, dtype=float)
    center = np.asarray(geometry_state['center'], dtype=float)
    simplex_dirs = np.asarray(geometry_state['simplex_dirs'], dtype=float)
    anchor_radius = np.asarray(geometry_state['anchor_radius'], dtype=float)
    anchor_points = center[None, :] + anchor_radius[:, None] * simplex_dirs

    distances = 0.5 * (
        np.sum((z1[:, None, :] - anchor_points[None, :, :]) ** 2, axis=2)
        + np.sum((z2[:, None, :] - anchor_points[None, :, :]) ** 2, axis=2)
    )
    assignments = np.argmin(distances, axis=1)
    class_frequency = np.bincount(assignments, minlength=anchor_points.shape[0]).astype(float)
    class_frequency = class_frequency / max(len(assignments), 1)

    mean_label = np.mean(pred_label, axis=0)
    frequent_classes = np.flatnonzero(mean_label > gpnf_threshold).astype(int)
    if 0 not in frequent_classes:
        frequent_classes = np.insert(frequent_classes, 0, 0)

    geometry_score = np.min(distances[:, frequent_classes], axis=1)
    return geometry_score, {
        'geometry_distance': distances,
        'geometry_assignment': assignments,
        'geometry_class_frequency': class_frequency,
        'geometry_frequent_classes': frequent_classes,
        'geometry_score_center': distances[:, 0],
    }


def compute_multiclass_geometry_anomaly_score(
    input,
    pred,
    pred_label,
    z1,
    z2,
    geometry_state,
    threshold=0.05,
    geometry_score_ratio=0.0,
    gpnf_threshold=0.05,
):
    B, W, D = input.shape
    base_anomaly_score = compute_base_anomaly_score(input, pred, pred_label, threshold=threshold).reshape(-1)
    geometry_raw, details = simpa_geometry_score(
        z1,
        z2,
        pred_label,
        geometry_state,
        gpnf_threshold=gpnf_threshold,
    )
    if geometry_raw is None:
        return base_anomaly_score, {'base_anomaly_score': base_anomaly_score, 'geometry_type': 'simpa'}

    geometry_smoothed = smooth_score(geometry_raw, w=int(W / 2))
    geometry_norm_stats = fit_minmax(geometry_smoothed)
    geometry_score = apply_minmax(geometry_smoothed, geometry_norm_stats, clip=False).reshape(-1)
    geometry_score_ratio = float(np.clip(geometry_score_ratio, 0.0, 1.0))
    anomaly_score = (1.0 - geometry_score_ratio) * base_anomaly_score + geometry_score_ratio * geometry_score
    details.update({
        'geometry_type': 'simpa',
        'base_anomaly_score': base_anomaly_score,
        'geometry_score': geometry_score,
        'geometry_score_raw': geometry_raw,
        'geometry_norm_stats': geometry_norm_stats,
        'geometry_score_ratio': geometry_score_ratio,
        'gpnf_threshold': float(gpnf_threshold),
    })
    return anomaly_score, details


def compute_selected_anomaly_score(
    test_inputs,
    test_prediction,
    test_pred_label,
    test_z1,
    test_z2,
    geometry_state,
    args,
):
    if geometry_state is not None and geometry_state.get('geometry_type') == 'simpa':
        return compute_multiclass_geometry_anomaly_score(
            test_inputs,
            test_prediction,
            test_pred_label,
            test_z1,
            test_z2,
            geometry_state,
            threshold=args.pnf_threshold,
            geometry_score_ratio=args.geometry_score_ratio,
            gpnf_threshold=args.gpnf_threshold,
        )

    return (
        compute_base_anomaly_score(
            test_inputs,
            test_prediction,
            test_pred_label,
            threshold=args.pnf_threshold,
        ),
        None,
    )


def parse_entities_arg(entities):
    if entities is None:
        return []
    entities = str(entities).strip()
    if entities == '' or entities.lower() in ['0', 'default', 'none']:
        return []
    return [entity.strip() for entity in entities.split(',') if entity.strip()]


def resolve_entity_list(entities_arg, default_entities, all_entities, aliases=None, normalizer=None):
    selected = parse_entities_arg(entities_arg)
    if not selected:
        return default_entities

    aliases = aliases or {}
    if len(selected) == 1 and selected[0].lower() in aliases:
        return aliases[selected[0].lower()]

    resolved = []
    valid = set(all_entities + default_entities)
    for entity in selected:
        candidate = normalizer(entity) if normalizer else entity
        if candidate not in valid:
            raise ValueError(
                f"Unknown entity '{entity}'. Use one of: {', '.join(all_entities[:10])}"
                + (" ..." if len(all_entities) > 10 else "")
            )
        resolved.append(candidate)
    return resolved


def normalize_smd_entity(entity):
    return entity if entity.startswith('machine-') else f'machine-{entity}'


def test_anomaly_score_exists(test_save_dir):
    return os.path.isfile(f'{test_save_dir}/anomaly_score.npy')


def evaluate_and_save_metric(test_save_dir, test_dataloader, anomaly_score, args, entity, split_name):
    metric_window = max(args.window_size // 2, 1)
    point_labels, point_scores = align_window_scores_to_point_labels(
        np.asarray(anomaly_score).reshape(-1),
        test_dataloader.dataset,
        args.window_size,
    )
    metric = compute_metrics(
        labels=point_labels,
        scores=point_scores,
        metric_window=metric_window,
    )
    metric.update({
        'dataset': args.dataset,
        'entity': entity,
        'split': split_name,
    })

    os.makedirs(test_save_dir, exist_ok=True)
    with open(f'{test_save_dir}/metric.json', 'w', encoding='utf-8') as f:
        json.dump(metric, f, indent=2, sort_keys=True)
    return metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--dataset', type=str, default='smd', choices=['iops', 'smd', 'smap', 'msl', 'yahoo', 'nab', 'genesis'], help='The dataset name')
    parser.add_argument('--entities', type=str, default='default', help='Comma-separated subset entities. Use default/global/all where supported.')

    parser.add_argument('--downsampling', type=int, default=1, help='The downsampling factor.')
    parser.add_argument('--batch-size', type=int, default=None, help='The batch size. Defaults to the dataset preset when omitted.')
    parser.add_argument('--window-size', type=int, default=100, help='The window size (defaults to 64)')
    parser.add_argument('--window-step', type=int, default=1, help='The sliding window (defaults to 1)')

    # Learning
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')

    # Model
    parser.add_argument('--anomaly-types', type=str, default='normal,spike,flip,speedup,noise,cutoff,average,scale,wander,contextual,upsidedown,mixture', help='List of anomaly types')

    # Architecture
    parser.add_argument('--embedding_dim', type=int, default=128, help='The size of embedding')
    parser.add_argument('--c-loss-ratio', type=float, default=0.1, help='Cross-entropy loss ratio for multiclass pseudo-anomaly labels.')
    parser.add_argument('--min_features', type=int, default=1, help='The minimum number of augmented features')
    parser.add_argument('--max_features', type=int, default=1, help='The maximum number of augmented features')
    parser.add_argument('--min_range', type=int, default=1, help='The range of inserted anomaly')

    parser.add_argument('--ae-mask-ratio', type=float, default=0.15, help='Random input mask ratio for masked/denoising ConvAE')
    parser.add_argument('--ae-noise-std', type=float, default=0.05, help='Gaussian input noise std for masked/denoising ConvAE')
    parser.add_argument('--sc-loss-ratio', type=float, default=0.001, help='SC geometry loss ratio.')
    parser.add_argument('--sc-radius-scale', type=float, default=0.25, help='Scale for pseudo-anomaly class anchor radii.')
    parser.add_argument('--eps-center', type=float, default=1e-3, help='Small-value clamp for the geometry center')
    parser.add_argument('--anchor-radius-base-quantile', type=float, default=0.75, help='Quantile used for class-level severity anchor base')
    parser.add_argument('--geometry-score-ratio', type=float, default=0.0, help='Fusion ratio for geometry score in the final anomaly score.')
    parser.add_argument('--gpnf-threshold', type=float, default=0.05, help='Threshold for geometry pseudo-normal filtering')
    parser.add_argument('--head-init-noise-std', type=float, default=1e-3, help='Independent noise std for projection head identity init')
    parser.add_argument('--pnf-threshold', type=float, default=0.05, help='Threshold for pseudo-normal filtering in anomaly scoring')
    parser.add_argument('--force-test', action='store_true', help='Rerun test inference even when saved test outputs already exist')

    # Computer
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference')
    parser.add_argument('--seed', type=int, default=0, help='The random seed')
    parser.add_argument('--run_name', type=str, default='test', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')

    args = parser.parse_args()
    print("Arguments:", str(args))

    device = utils.init_dl_program(args.gpu, seed=args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print('Device', device)

    args.anomaly_types = args.anomaly_types.split(',') if args.anomaly_types else ['normal','spike','flip','speedup','noise','cutoff','average','scale','wander','contextual','upsidedown','mixture']

    if args.dataset == 'iops':
        args.n_features = 1
        min_features = 1
        max_features = 1
        args.batch_size = args.batch_size or 128
        args.window_size = 100
        args.window_step = 10
        all_entities = ['KPI-05f10d3a-239c-3bef-9bdc-a2feeb0037aa', 'KPI-0efb375b-b902-3661-ab23-9a0bb799f4e3', 'KPI-1c6d7a26-1f1a-3321-bb4d-7a9d969ec8f0', 'KPI-301c70d8-1630-35ac-8f96-bc1b6f4359ea', 'KPI-42d6616d-c9c5-370a-a8ba-17ead74f3114', 'KPI-43115f2a-baeb-3b01-96f7-4ea14188343c', 'KPI-431a8542-c468-3988-a508-3afd06a218da', 'KPI-4d2af31a-9916-3d9f-8a8e-8a268a48c095', 'KPI-54350a12-7a9d-3ca8-b81f-f886b9d156fd', 'KPI-55f8b8b8-b659-38df-b3df-e4a5a8a54bc9', 'KPI-57051487-3a40-3828-9084-a12f7f23ee38', 'KPI-6a757df4-95e5-3357-8406-165e2bd49360', 'KPI-6d1114ae-be04-3c46-b5aa-be1a003a57cd', 'KPI-6efa3a07-4544-34a0-b921-a155bd1a05e8', 'KPI-7103fa0f-cac4-314f-addc-866190247439', 'KPI-847e8ecc-f8d2-3a93-9107-f367a0aab37d', 'KPI-8723f0fb-eaef-32e6-b372-6034c9c04b80', 'KPI-9c639a46-34c8-39bc-aaf0-9144b37adfc8', 'KPI-a07ac296-de40-3a7c-8df3-91f642cc14d0', 'KPI-a8c06b47-cc41-3738-9110-12df0ee4c721', 'KPI-ab216663-dcc2-3a24-b1ee-2c3e550e06c9', 'KPI-adb2fde9-8589-3f5b-a410-5fe14386c7af', 'KPI-ba5f3328-9f3f-3ff5-a683-84437d16d554', 'KPI-c02607e8-7399-3dde-9d28-8a8da5e5d251', 'KPI-c69a50cf-ee03-3bd7-831e-407d36c7ee91', 'KPI-da10a69f-d836-3baa-ad40-3e548ecf1fbd', 'KPI-e0747cad-8dc8-38a9-a9ab-855b61f5551d', 'KPI-f0932edd-6400-3e63-9559-0a9860a1baa9', 'KPI-ffb82d38-5f00-37db-abc0-5d2e4e4cb6aa']
        entity_list = resolve_entity_list(
            args.entities,
            default_entities=all_entities,
            all_entities=all_entities,
            aliases={'all': all_entities},
        )
    elif args.dataset == 'yahoo':
        args.n_features = 1
        min_features = 1
        max_features = 1
        args.batch_size = args.batch_size or 256
        args.window_size = 100
        args.window_step = 10
        all_entities = list_yahoo_entities(root_dir='./dataset')
        entity_list = resolve_entity_list(
            args.entities,
            default_entities=[all_entities[0]],
            all_entities=all_entities,
            aliases={'all': all_entities},
        )
    elif args.dataset == 'nab':
        args.n_features = 1
        min_features = 1
        max_features = 1
        args.batch_size = args.batch_size or 256
        args.window_size = 100
        args.window_step = 10
        all_entities = list_nab_entities(root_dir='./dataset')
        entity_list = resolve_entity_list(
            args.entities,
            default_entities=[all_entities[0]],
            all_entities=all_entities,
            aliases={'all': all_entities},
        )
    elif args.dataset == 'genesis':
        args.n_features = 18
        min_features = 1
        max_features = args.n_features
        args.batch_size = args.batch_size or 128
        args.window_size = 100
        args.window_step = 10
        entity_list = resolve_entity_list(
            args.entities,
            default_entities=['genesis'],
            all_entities=['genesis'],
            aliases={'all': ['genesis']},
        )
    elif args.dataset == 'smd':
        args.n_features = 38
        min_features = 1
        max_features = args.n_features
        args.batch_size = args.batch_size or 128
        args.window_size = 100
        args.window_step = 10
        all_entities = ["1-1","1-2","1-3","1-4","1-5","1-6","1-7","1-8","2-1","2-2","2-3","2-4","2-5","2-6","2-7","2-8","2-9","3-1","3-2","3-3","3-4","3-5","3-6","3-7","3-8","3-9","3-10","3-11"]
        all_entities = [f'machine-{entity}' for entity in all_entities]
        entity_list = resolve_entity_list(
            args.entities,
            default_entities=all_entities,
            all_entities=all_entities,
            aliases={'all': all_entities},
            normalizer=normalize_smd_entity,
        )
    elif args.dataset == 'smap':
        args.n_features = 25
        min_features = 1
        max_features = args.n_features
        args.batch_size = args.batch_size or 128
        args.window_size = 100
        args.window_step = 10
        all_entities = ['A-1', 'A-2', 'A-3', 'A-4', 'A-7', 'B-1', 'D-1', 'D-11', 'D-13', 'D-2', 'D-3', 'D-4', 'D-5', 'D-6', 'D-7', 'D-8', 'D-9', 'E-1', 'E-10', 'E-11', 'E-12', 'E-13', 'E-2', 'E-3', 'E-4', 'E-5', 'E-6', 'E-7', 'E-8', 'E-9', 'F-1', 'F-2', 'F-3', 'G-1', 'G-2', 'G-3', 'G-4', 'G-6', 'G-7', 'P-1', 'P-2', 'P-2', 'P-3', 'P-4', 'P-7', 'R-1', 'S-1', 'T-1', 'T-2', 'T-3']
        entity_list = resolve_entity_list(
            args.entities,
            default_entities=['smap'],
            all_entities=all_entities,
            aliases={'all': all_entities, 'global': ['smap']},
        )
    elif args.dataset == 'msl':
        args.n_features = 55
        min_features = 1
        max_features = args.n_features
        args.batch_size = args.batch_size or 128
        args.window_size = 100
        args.window_step = 10
        all_entities = ['C-1', 'D-14', 'D-15', 'D-16', 'F-4', 'F-5', 'F-7', 'F-8', 'M-1', 'M-2', 'M-3', 'M-4', 'M-5', 'M-6', 'M-7', 'P-10', 'P-11', 'P-14', 'P-15', 'T-12', 'T-13', 'T-4', 'T-5']
        entity_list = resolve_entity_list(
            args.entities,
            default_entities=['msl'],
            all_entities=all_entities,
            aliases={'all': all_entities, 'global': ['msl']},
        )


    print('Entities:', entity_list)

    for entity in entity_list:
        params = utils.AttrDict(
            batch_size=args.batch_size,
            lr=args.lr,
            epoch=100,
            max_grad_norm=1.0,
            seed=args.seed,
            )
        params.override(model_parameters(args))

        dataparams = utils.AttrDict(
            dataset=args.dataset,
            entities=entity,
            downsampling=args.downsampling,
            batch_size=args.batch_size,
            window_size=args.window_size,
            window_step=args.window_step,
            anomaly_types=args.anomaly_types,
            min_range=args.min_range,
            min_features=min_features,
            max_features=max_features,
            anchor_radius_base_quantile=args.anchor_radius_base_quantile,
            sc_radius_scale=args.sc_radius_scale,
        )


        base_dir = f'./result/{args.run_name}'
        data_dir = f'{args.dataset}/{entity}/d{dataparams.downsampling}_b{dataparams.batch_size}_w{dataparams.window_size}_s{dataparams.window_step}'
        model_dir = f'{base_dir}/{data_dir}/{args.seed}'

        if test_anomaly_score_exists(f'{model_dir}/test_all'):
            print(f'{model_dir}/test_all/anomaly_score.npy', 'exists')
            # continue


        train_dataloader, val_dataloader = datautils.load_dataloader_aug(dataparams, group='train')
        test_dataloader = datautils.load_dataloader_aug(dataparams, anomaly_types=['normal'], anomaly_types_for_dict=args.anomaly_types, group='test_all')
        print('# of train',len(train_dataloader))
        print('# of valid',len(val_dataloader) if val_dataloader else None)
        print('# of test', len(test_dataloader))

        args.Train=True
        if test_anomaly_score_exists(f'{model_dir}/test_all'):
            print(f'{model_dir}/test_all/anomaly_score.npy', 'exists')
            args.Train=False
        if args.Train:
            print('Train')
            model = MulticlassAnomalyTrainer(model_dir = model_dir, params = params, device = device)
            model.train(train_dataloader, val_dataloader)


        args.Test=True
        test_save_dir = f'{model_dir}/test_all'
        if test_anomaly_score_exists(test_save_dir) and not args.force_test:
            print(f'{test_save_dir}/anomaly_score.npy', 'exists')
            args.Test=False
        if args.Test:
            print('Test:Test_all')
            test_inputs, test_prediction, test_pred_label, test_z1, test_z2, geometry_state = test(test_dataloader, model_dir, params, device)
            anomaly_score, _ = compute_selected_anomaly_score(
                test_inputs,
                test_prediction,
                test_pred_label,
                test_z1,
                test_z2,
                geometry_state,
                args,
            )
            os.makedirs(test_save_dir, exist_ok=True)
            np.save(f'{test_save_dir}/anomaly_score.npy',anomaly_score)

        if os.path.isfile(f'{test_save_dir}/anomaly_score.npy'):
            anomaly_score = np.load(f'{test_save_dir}/anomaly_score.npy')
            evaluate_and_save_metric(
                test_save_dir,
                test_dataloader,
                anomaly_score,
                args,
                entity,
                split_name='test_all',
            )

        if entity in ['smap','msl']:
            if entity=='smap': each_entity_list = ['A-1', 'A-2', 'A-3', 'A-4', 'A-7', 'B-1', 'D-1', 'D-11', 'D-13', 'D-2', 'D-3', 'D-4', 'D-5', 'D-6', 'D-7', 'D-8', 'D-9', 'E-1', 'E-10', 'E-11', 'E-12', 'E-13', 'E-2', 'E-3', 'E-4', 'E-5', 'E-6', 'E-7', 'E-8', 'E-9', 'F-1', 'F-2', 'F-3', 'G-1', 'G-2', 'G-3', 'G-4', 'G-6', 'G-7', 'P-1', 'P-2', 'P-2', 'P-3', 'P-4', 'P-7', 'R-1', 'S-1', 'T-1', 'T-2', 'T-3']
            if entity=='msl': each_entity_list = ['C-1', 'D-14', 'D-15', 'D-16', 'F-4', 'F-5', 'F-7', 'F-8', 'M-1', 'M-2', 'M-3', 'M-4', 'M-5', 'M-6', 'M-7', 'P-10', 'P-11', 'P-14', 'P-15', 'T-12', 'T-13', 'T-4', 'T-5']
            for ent in each_entity_list:
                dataparams.entities=ent
                test_dataloader = datautils.load_dataloader_aug(dataparams, anomaly_types=['normal'], anomaly_types_for_dict=args.anomaly_types, group='test_all')
                args.Test=True
                test_save_dir = f'{model_dir}/test_each/{ent}/test_all'
                if test_anomaly_score_exists(test_save_dir) and not args.force_test:
                    print(f'{test_save_dir}/anomaly_score.npy', 'exists')
                    args.Test=False
                if args.Test:
                    print('Test:Test_all')
                    test_inputs, test_prediction, test_pred_label, test_z1, test_z2, geometry_state = test(test_dataloader, model_dir, params, device)
                    anomaly_score, _ = compute_selected_anomaly_score(
                        test_inputs,
                        test_prediction,
                        test_pred_label,
                        test_z1,
                        test_z2,
                        geometry_state,
                        args,
                    )
                    os.makedirs(test_save_dir, exist_ok=True)
                    np.save(f'{test_save_dir}/anomaly_score.npy',anomaly_score)
                if os.path.isfile(f'{test_save_dir}/anomaly_score.npy'):
                    anomaly_score = np.load(f'{test_save_dir}/anomaly_score.npy')
                    evaluate_and_save_metric(
                        test_save_dir,
                        test_dataloader,
                        anomaly_score,
                        args,
                        ent,
                        split_name='test_each',
                    )

        print('Finish')
