import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import List, Optional, Tuple, Union
from pathlib import Path

from .dataset import Entity, Dataset


MACHINES = ['machine-1-1','machine-1-2','machine-1-3','machine-1-4','machine-1-5','machine-1-6','machine-1-7','machine-1-8',
            'machine-2-1', 'machine-2-2','machine-2-3','machine-2-4','machine-2-5','machine-2-6','machine-2-7','machine-2-8','machine-2-9',
            'machine-3-1', 'machine-3-2', 'machine-3-3', 'machine-3-4','machine-3-5','machine-3-6','machine-3-7','machine-3-8', 'machine-3-9',
            'machine-3-10', 'machine-3-11']
smap_data_set_number = ['A-1', 'A-2', 'A-3', 'A-4', 'A-7', 'B-1', 'D-1', 'D-11', 'D-13', 'D-2', 'D-3', 'D-4', 'D-5', 'D-6', 'D-7', 'D-8', 'D-9', 'E-1', 'E-10', 'E-11', 'E-12', 'E-13', 'E-2', 'E-3', 'E-4', 'E-5', 'E-6', 'E-7', 'E-8', 'E-9', 'F-1', 'F-2', 'F-3', 'G-1', 'G-2', 'G-3', 'G-4', 'G-6', 'G-7', 'P-1', 'P-2', 'P-2', 'P-3', 'P-4', 'P-7', 'R-1', 'S-1', 'T-1', 'T-2', 'T-3']
msl_data_set_number = ['C-1', 'D-14', 'D-15', 'D-16', 'F-4', 'F-5', 'F-7', 'F-8', 'M-1', 'M-2', 'M-3', 'M-4', 'M-5', 'M-6', 'M-7', 'P-10', 'P-11', 'P-14', 'P-15', 'T-12', 'T-13', 'T-4', 'T-5']


def load_data(dataset: str, group: str, entities: Union[str, List[str]], downsampling: float=None, min_length: float=None, root_dir:str='./data', normalize:bool=True, verbose:bool=True, validation:bool=False):
    """Function to load TS anomaly detection datasets.
    Parameters
    ----------
    dataset: str
        Name of the dataset.
    group: str
        The train or test split.
    entities: Union[str, List[str]]
        Entities to load from the dataset.
    downsampling: Optional[float]
        Whether and the extent to downsample the data.
    root_dir: str
        Path to the directory where the datasets are stored.
    normalize: bool
        Whether to normalize Y.
    verbose: bool
        Controls verbosity
    """
    if dataset == 'smd':
        return load_smd(group=group, machines=entities, downsampling=downsampling, root_dir=root_dir, normalize=normalize, verbose=verbose, validation=validation)
    elif dataset == 'msl':
        if entities=='msl':
            entities=msl_data_set_number
        return load_msl(group=group, channels=entities, downsampling=downsampling, root_dir=root_dir, normalize=normalize, verbose=verbose, validation=validation)
    elif dataset == 'smap':
        if entities=='smap':
            entities=smap_data_set_number
        return load_smap(group=group, channels=entities, downsampling=downsampling, root_dir=root_dir, normalize=normalize, verbose=verbose, validation=validation)
    elif dataset == 'iops':
        return load_iops(group=group, filename=entities, downsampling=downsampling, root_dir=root_dir, normalize=normalize, verbose=verbose, validation=validation)
    elif dataset == 'yahoo':
        return load_yahoo(group=group, entities=entities, downsampling=downsampling, root_dir=root_dir, normalize=normalize, verbose=verbose, validation=validation)
    elif dataset == 'nab':
        return load_nab(group=group, entities=entities, downsampling=downsampling, root_dir=root_dir, normalize=normalize, verbose=verbose, validation=validation)
    elif dataset == 'genesis':
        return load_genesis(group=group, downsampling=downsampling, root_dir=root_dir, normalize=normalize, verbose=verbose, validation=validation)

def load_smd(group, machines=None, downsampling=None, root_dir='./data', normalize=True, verbose=True, validation=False):
    if machines is None:
        machines = MACHINES

    if isinstance(machines, str):
        machines = [machines]

    root_dir = f'{root_dir}/ServerMachineDataset'

    if group=='train':
        entities, entities_val = [], []
        for machine in machines:
            name = 'smd-train'
            name_val = 'smd-val'
            train_file = f'{root_dir}/train/{machine}.txt'
            Y = np.loadtxt(train_file, delimiter=',').T

            if downsampling is not None:
                n_features, n_t = Y.shape

                right_padding = downsampling - n_t%downsampling
                Y = np.pad(Y, ((0,0), (right_padding, 0) ))

                Y = Y.reshape(n_features, Y.shape[-1]//downsampling, downsampling).max(axis=2)

            if validation:
                train_length = int(Y.shape[1]*0.9)
                entity = Entity(Y=Y[:, :train_length], name=machine, verbose=verbose)
                entities.append(entity)
                entity_val = Entity(Y=Y[:, train_length:], name=machine, verbose=verbose)
                entities_val.append(entity_val)
            else:
                entity = Entity(Y=Y, name=machine, verbose=verbose)
                entities.append(entity)

        if validation:
            smd = Dataset(entities=entities, name=name, verbose=verbose)
            smd_val = Dataset(entities=entities_val, name=name_val, verbose=verbose)
            return smd, smd_val
        else:
            smd = Dataset(entities=entities, name=name, verbose=verbose)
            return smd

    elif group=='test':
        entities = []
        for machine in machines:
            name = 'smd-test'
            test_file = f'{root_dir}/test/{machine}.txt'
            label_file = f'{root_dir}/test_label/{machine}.txt'

            Y = np.loadtxt(test_file, delimiter=',').T
            labels = np.loadtxt(label_file, delimiter=',')

            if downsampling is not None:
                n_features, n_t = Y.shape
                right_padding = downsampling - n_t%downsampling

                Y = np.pad(Y, ((0,0), (right_padding, 0) ))
                labels = np.pad(labels, (right_padding, 0))

                Y = Y.reshape(n_features, Y.shape[-1]//downsampling, downsampling).max(axis=2)
                labels = labels.reshape(labels.shape[0]//downsampling, downsampling).max(axis=1)

            labels = labels[None, :]
            entity = Entity(Y=Y, name=machine, labels=labels, verbose=verbose)
            entities.append(entity)

        smd = Dataset(entities=entities, name=name, verbose=verbose)
        return smd


def load_msl(group, channels=None, downsampling=None, root_dir='./data', normalize=True, verbose=True, validation=False):
    return _load_nasa(group=group, spacecraft='MSL', channels=channels, downsampling=downsampling, root_dir=root_dir, normalize=normalize, verbose=verbose, validation=validation)

def load_smap(group, channels=None, downsampling=None, root_dir='./data', normalize=True, verbose=True, validation=False):
    return _load_nasa(group=group, spacecraft='SMAP', channels=channels, downsampling=downsampling, root_dir=root_dir, normalize=normalize, verbose=verbose, validation=validation)

def _load_nasa(group, spacecraft, channels=None, downsampling=None, root_dir='./data', normalize=True, verbose=True, validation=False):
    root_dir = f'{root_dir}/NASA'
    meta_data = pd.read_csv(f'{root_dir}/labeled_anomalies.csv')

    CHANNEL_IDS =  list(meta_data.loc[meta_data['spacecraft'] == spacecraft]['chan_id'].values)
    if verbose:
        print(f'Number of Entities: {len(CHANNEL_IDS)}')

    print('channels',channels)
    print('CHANNELS',sorted(CHANNEL_IDS))
    if channels is None: channels = CHANNEL_IDS

    if isinstance(channels, str):
        channels = [channels]

    if group == 'train':
        entities, entities_val = [], []
        for channel_id in channels:
            if normalize:
                with open(f'{root_dir}/train/{channel_id}.npy', 'rb') as f:
                    Y = np.load(f)
                scaler = MinMaxScaler()
                scaler.fit(Y)

            name = f'{spacecraft}-train'
            name_val = f'{spacecraft}-val'
            with open(f'{root_dir}/train/{channel_id}.npy', 'rb') as f:
                Y = np.load(f).T

            if normalize:
                Y = scaler.transform(Y.T).T

            if downsampling is not None:
                n_features, n_t = Y.shape

                right_padding = downsampling - n_t%downsampling
                Y = np.pad(Y, ((0,0), (right_padding, 0) ))

                Y = Y.reshape(n_features, Y.shape[-1]//downsampling, downsampling).max(axis=2)


            if validation:
                train_length = int(Y.shape[1]*0.9)
                entity = Entity(Y=Y[:, :train_length], name=channel_id, verbose=verbose)
                entities.append(entity)
                entity_val = Entity(Y=Y[:, train_length:], name=channel_id, verbose=verbose)
                entities_val.append(entity_val)
            else:
                entity = Entity(Y=Y, name=channel_id, verbose=verbose)
                entities.append(entity)

        if validation:
            data = Dataset(entities=entities, name=name, verbose=verbose)
            data_val = Dataset(entities=entities_val, name=name_val, verbose=verbose)
            return data, data_val
        else:
            data = Dataset(entities=entities, name=name, verbose=verbose)
            return data


    elif group == 'test':
        entities = []
        for channel_id in channels:
            if normalize:
                with open(f'{root_dir}/train/{channel_id}.npy', 'rb') as f:
                    Y = np.load(f) 
                scaler = MinMaxScaler()
                scaler.fit(Y)

            name = f'{spacecraft}-test'
            with open(f'{root_dir}/test/{channel_id}.npy', 'rb') as f:
                Y = np.load(f).T 

            if normalize:
                Y = scaler.transform(Y.T).T

            labels = np.zeros(Y.shape[1])
            anomalous_sequences = eval(meta_data.loc[meta_data['chan_id'] == channel_id]['anomaly_sequences'].values[0])
            if verbose: print('Anomalous sequences:', anomalous_sequences)

            for interval in anomalous_sequences:
                labels[interval[0]:interval[1]] = 1

            if downsampling is not None:
                n_features, n_t = Y.shape
                right_padding = downsampling - n_t%downsampling

                Y = np.pad(Y, ((0,0), (right_padding, 0) ))
                labels = np.pad(labels, (right_padding, 0))

                Y = Y.reshape(n_features, Y.shape[-1]//downsampling, downsampling).max(axis=2)
                labels = labels.reshape(labels.shape[0]//downsampling, downsampling).max(axis=1)

            labels = labels[None, :]
            entity = Entity(Y=Y, name=channel_id, labels=labels, verbose=verbose)
            entities.append(entity)

        data = Dataset(entities=entities, name=name, verbose=verbose)
        return data

def load_genesis(group, downsampling=None, root_dir='./data', normalize=True, verbose=True, validation=False):
    csv_path = (
        Path(root_dir)
        / 'Genesis'
        / 'multivariate'
        / 'Genesis'
        / 'genesis-anomalies.test.csv'
    )

    df = pd.read_csv(csv_path)
    feature_cols = [
        col for col in df.columns
        if col not in {'timestamp', 'is_anomaly'}
    ]
    Y = df[feature_cols].to_numpy(dtype=np.float32).T
    labels = (df['is_anomaly'].to_numpy(dtype=np.float32) > 0).astype(np.float32)

    train_end = 3604
    if normalize:
        scaler = MinMaxScaler()
        scaler.fit(Y[:, :train_end].T)
        Y = scaler.transform(Y.T).T

    if group == 'train':
        Y_train = Y[:, :train_end]
        Y_train, _ = _downsample_series(Y_train, None, downsampling)
        if validation:
            val_length = max(int(Y_train.shape[1] * 0.1), 100)
            val_length = min(val_length, Y_train.shape[1] - 1)
            train_length = Y_train.shape[1] - val_length
            train_entity = Entity(Y=Y_train[:, :train_length], name='genesis', verbose=verbose)
            val_entity = Entity(Y=Y_train[:, train_length:], name='genesis', verbose=verbose)
            return (
                Dataset(entities=[train_entity], name='genesis-train', verbose=verbose),
                Dataset(entities=[val_entity], name='genesis-val', verbose=verbose),
            )
        entity = Entity(Y=Y_train, name='genesis', verbose=verbose)
        return Dataset(entities=[entity], name='genesis-train', verbose=verbose)

    if group == 'test':
        Y_test = Y[:, train_end:]
        labels_test = labels[train_end:]
        Y_test, labels_test = _downsample_series(Y_test, labels_test, downsampling)
        entity = Entity(
            Y=Y_test,
            name='genesis',
            labels=labels_test[None, :],
            verbose=verbose,
        )
        return Dataset(entities=[entity], name='genesis-test', verbose=verbose)


def _parse_univariate_filename(path: Path) -> dict:
    parts = path.stem.split('_')
    train_pos = parts.index('tr')
    first_pos = parts.index('1st')
    id_value = parts[parts.index('id') + 1] if 'id' in parts else parts[0]
    return {
        'prefix': parts[0],
        'dataset': parts[1],
        'id': id_value,
        'domain': '_'.join(parts[4:train_pos]) if train_pos > 4 else '',
        'train_end': int(parts[train_pos + 1]),
        'first_anomaly': int(parts[first_pos + 1]),
        'stem': path.stem,
    }


def _univariate_sort_key(path: Path) -> Tuple[int, str]:
    prefix = path.stem.split('_', 1)[0]
    try:
        return int(prefix), path.name
    except ValueError:
        return 10**9, path.name


def _univariate_files(dataset_name: str, data_dir: Path) -> List[Path]:
    files = [
        path for path in data_dir.rglob('*.csv')
        if len(path.stem.split('_')) > 1 and path.stem.split('_')[1] == dataset_name
    ]
    return sorted(files, key=_univariate_sort_key)


def list_yahoo_entities(root_dir='./data') -> List[str]:
    return [path.stem.split('_', 1)[0] for path in _univariate_files('YAHOO', Path(root_dir) / 'Yahoo')]


def list_nab_entities(root_dir='./data') -> List[str]:
    return [path.stem.split('_', 1)[0] for path in _univariate_files('NAB', Path(root_dir) / 'NAB')]


def _normalize_entity_arg(entities: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
    if entities is None:
        return None
    if isinstance(entities, str):
        return [item.strip() for item in entities.split(',') if item.strip()]
    return [str(item).strip() for item in entities if str(item).strip()]


def _select_univariate_files(dataset_name: str, data_dir: Path, entities: Optional[Union[str, List[str]]]) -> List[Path]:
    files = _univariate_files(dataset_name, data_dir)
    requested = _normalize_entity_arg(entities)
    if requested is None or any(item.lower() in {'all', dataset.lower(), dataset_name.lower()} for item in requested):
        return files

    requested_set = set(requested)
    selected = []
    for path in files:
        meta = _parse_univariate_filename(path)
        keys = {
            str(meta['prefix']),
            str(meta['id']),
            str(meta['stem']),
            path.name,
            f"id_{meta['id']}",
        }
        if keys.intersection(requested_set):
            selected.append(path)

    return selected


def _read_univariate_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    lower_cols = {str(col).lower(): col for col in df.columns}
    label_col = None
    for candidate in ('label', 'labels', 'anomaly', 'is_anomaly'):
        if candidate in lower_cols:
            label_col = lower_cols[candidate]
            break
    if label_col is None:
        binary_cols = [
            col for col in df.columns
            if set(pd.Series(df[col]).dropna().unique()).issubset({0, 1})
        ]
        label_col = binary_cols[-1]

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    value_cols = [
        col for col in numeric_cols
        if col != label_col and str(col).lower() not in {'timestamp', 'time', 'date'}
    ]

    Y = df[value_cols].to_numpy(dtype=np.float32).T
    labels = (df[label_col].to_numpy(dtype=np.float32) > 0).astype(np.float32)
    return Y, labels


def _downsample_series(Y: np.ndarray, labels: Optional[np.ndarray], downsampling: Optional[int]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if downsampling is None or downsampling <= 1:
        return Y, labels

    n_features, n_t = Y.shape
    right_padding = (-n_t) % downsampling
    if right_padding:
        Y = np.pad(Y, ((0, 0), (0, right_padding)))
        if labels is not None:
            labels = np.pad(labels, (0, right_padding))

    Y = Y.reshape(n_features, Y.shape[-1] // downsampling, downsampling).max(axis=2)
    if labels is not None:
        labels = labels.reshape(labels.shape[0] // downsampling, downsampling).max(axis=1)
    return Y, labels


def load_yahoo(group, entities=None, downsampling=None, root_dir='./data', normalize=True, verbose=True, validation=False):
    return _load_univariate_csv_dataset(
        group=group,
        dataset_name='YAHOO',
        data_dir=Path(root_dir) / 'Yahoo',
        entities=entities,
        downsampling=downsampling,
        normalize=normalize,
        verbose=verbose,
        validation=validation,
    )


def load_nab(group, entities=None, downsampling=None, root_dir='./data', normalize=True, verbose=True, validation=False):
    return _load_univariate_csv_dataset(
        group=group,
        dataset_name='NAB',
        data_dir=Path(root_dir) / 'NAB',
        entities=entities,
        downsampling=downsampling,
        normalize=normalize,
        verbose=verbose,
        validation=validation,
    )


def _load_univariate_csv_dataset(group, dataset_name, data_dir, entities=None, downsampling=None, normalize=True, verbose=True, validation=False):
    files = _select_univariate_files(dataset_name=dataset_name, data_dir=data_dir, entities=entities)
    if verbose:
        print(f'Number of {dataset_name} datasets: {len(files)}')

    loaded_entities, loaded_entities_val = [], []
    name = f'{dataset_name.lower()}-{group}'
    name_val = f'{dataset_name.lower()}-val'

    for path in files:
        meta = _parse_univariate_filename(path)
        Y, labels = _read_univariate_csv(path)
        train_end = int(meta['train_end'])
        train_end = min(max(train_end, 1), Y.shape[1] - 1)

        if normalize:
            scaler = MinMaxScaler()
            scaler.fit(Y[:, :train_end].T)
            Y = scaler.transform(Y.T).T

        if group == 'train':
            Y_train = Y[:, :train_end]
            Y_train, _ = _downsample_series(Y_train, None, downsampling)
            if validation:
                if Y_train.shape[1] >= 200:
                    val_length = max(int(Y_train.shape[1] * 0.1), 100)
                else:
                    val_length = max(int(Y_train.shape[1] * 0.1), 1)
                val_length = min(val_length, Y_train.shape[1] - 1)
                train_length = Y_train.shape[1] - val_length
                loaded_entities.append(Entity(Y=Y_train[:, :train_length], name=str(meta['stem']), verbose=verbose))
                loaded_entities_val.append(Entity(Y=Y_train[:, train_length:], name=str(meta['stem']), verbose=verbose))
            else:
                loaded_entities.append(Entity(Y=Y_train, name=str(meta['stem']), verbose=verbose))
        elif group == 'test':
            Y_test = Y[:, train_end:]
            labels_test = labels[train_end:]
            Y_test, labels_test = _downsample_series(Y_test, labels_test, downsampling)
            loaded_entities.append(
                Entity(
                    Y=Y_test,
                    name=str(meta['stem']),
                    labels=labels_test[None, :],
                    verbose=verbose,
                )
            )

    if validation:
        data = Dataset(entities=loaded_entities, name=name, verbose=verbose)
        data_val = Dataset(entities=loaded_entities_val, name=name_val, verbose=verbose)
        return data, data_val
    data = Dataset(entities=loaded_entities, name=name, verbose=verbose)
    return data


def load_iops(group, filename, downsampling=None, root_dir='./data', normalize=True, verbose=True, validation=False):
    root_dir = f'{root_dir}/IOPS/{filename}'

    if group == 'train':
        df = pd.read_csv(f'{root_dir}.train.out', header=None, names=['Value', 'Label'])
        Y = np.array(df['Value']).reshape(1,-1)

        name = f'{filename}-train'
        name_val = f'{filename}-val'
        if normalize:
            scaler = MinMaxScaler()
            scaler.fit(Y.T)
            Y = scaler.transform(Y.T).T
        if downsampling is not None:
            n_features, n_t = Y.shape
            right_padding = downsampling - n_t%downsampling
            Y = np.pad(Y, ((0,0), (right_padding, 0) ))
            Y = Y.reshape(n_features, Y.shape[-1]//downsampling, downsampling).max(axis=2)


        if validation:
            train_length = int(Y.shape[1]*0.9)
            entity = Entity(Y=Y[:, :train_length], name=name, verbose=verbose)
            entity_val = Entity(Y=Y[:, train_length:], name=name_val, verbose=verbose)
            data = Dataset(entities=[entity], name=name, verbose=verbose)
            data_val = Dataset(entities=[entity_val], name=name_val, verbose=verbose)
            return data, data_val
        else:
            entity = Entity(Y=Y, name=name, verbose=verbose)
            data = Dataset(entities=[entity], name=name, verbose=verbose)
            return data


    elif group == 'test':
        df = pd.read_csv(f'{root_dir}.test.out', header=None, names=['Value', 'Label'])
        Y = np.array(df['Value']).reshape(1,-1)
        if normalize:
            df_train = pd.read_csv(f'{root_dir}.train.out', header=None, names=['Value', 'Label'])
            Y_train = np.array(df_train['Value']).reshape(1,-1)
            scaler = MinMaxScaler()
            scaler.fit(Y_train.T)
            Y = scaler.transform(Y.T).T

        name = f'{filename}-test'
        labels = np.array(df['Label'])

        if downsampling is not None:
            n_features, n_t = Y.shape
            right_padding = downsampling - n_t%downsampling

            Y = np.pad(Y, ((0,0), (right_padding, 0) ))
            labels = np.pad(labels, (right_padding, 0))

            Y = Y.reshape(n_features, Y.shape[-1]//downsampling, downsampling).max(axis=2)
            labels = labels.reshape(labels.shape[0]//downsampling, downsampling).max(axis=1)

        labels = labels[None, :]
        entity = Entity(Y=Y, name=name, labels=labels, verbose=verbose)
        data = Dataset(entities=[entity], name=name, verbose=verbose)
        return data
