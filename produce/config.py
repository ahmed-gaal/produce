from pathlib import Path

class Config:
    random_seed = 42
    assets_path = Path('./assets')
    original_dataset_path = assets_path / 'original_dataset' / 'dataset1.csv'
    dataset_path = assets_path / 'data_set'
    features_path = assets_path / 'feature'
    models_path = assets_path / 'model'
    metrics_file_path = assets_path / 'metric.json'