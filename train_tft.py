"""
Training Pipeline para Temporal Fusion Transformer (TFT)
Predice probabilidad de brote en horizontes de 7, 14, 30 días
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import logging

import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import mlflow
import mlflow.pytorch

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import asyncpg

logger = logging.getLogger(__name__)


class MSRelapsePredictorDataset:
    """
    Dataset builder para predicción de brotes de EM
    """
    
    def __init__(self, db_dsn: str):
        self.db_dsn = db_dsn
        
    async def load_data(
        self,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Carga datos de features + clinical events desde DB
        Construye dataset supervisado para training
        """
        conn = await asyncpg.connect(self.db_dsn)
        
        try:
            # Load feature windows (X)
            features_query = """
                SELECT 
                    fw.user_id_hash,
                    fw.window_end as date,
                    fw.window_size_days,
                    fw.features,
                    fw.num_datapoints
                FROM feature_windows fw
                JOIN users u ON fw.user_id_hash = u.user_id_hash
                WHERE u.status = 'active'
                    AND fw.window_end >= $1
                    AND fw.window_end <= $2
                ORDER BY fw.user_id_hash, fw.window_end
            """
            
            start = start_date or '2024-01-01'
            end = end_date or datetime.now().strftime('%Y-%m-%d')
            
            rows = await conn.fetch(features_query, start, end)
            
            # Parse JSONB features
            data = []
            for row in rows:
                feat = dict(row['features'])
                feat['user_id_hash'] = row['user_id_hash']
                feat['date'] = row['date']
                feat['window_size_days'] = row['window_size_days']
                data.append(feat)
            
            df = pd.DataFrame(data)
            
            # Load clinical events (Y - ground truth)
            events_query = """
                SELECT 
                    user_id_hash,
                    event_date,
                    event_type
                FROM clinical_events
                WHERE event_type = 'relapse'
                    AND event_date >= $1
                    AND event_date <= $2
            """
            
            event_rows = await conn.fetch(events_query, start, end)
            events_df = pd.DataFrame([dict(r) for r in event_rows])
            
            # Create labels for different horizons
            df = self._create_labels(df, events_df, horizons=[7, 14, 30])
            
            logger.info(f"Loaded {len(df)} feature windows, {len(events_df)} clinical events")
            
            return df
            
        finally:
            await conn.close()
    
    def _create_labels(
        self,
        features_df: pd.DataFrame,
        events_df: pd.DataFrame,
        horizons: List[int]
    ) -> pd.DataFrame:
        """
        Crea labels binarias: ¿habrá un brote en los próximos N días?
        """
        df = features_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        if events_df.empty:
            for h in horizons:
                df[f'relapse_in_{h}d'] = 0
            return df
        
        events_df['event_date'] = pd.to_datetime(events_df['event_date'])
        
        # For each feature window, check if relapse occurs in horizon
        for h in horizons:
            labels = []
            
            for _, row in df.iterrows():
                user_id = row['user_id_hash']
                date = row['date']
                
                # Check if any event for this user in next h days
                user_events = events_df[
                    (events_df['user_id_hash'] == user_id) &
                    (events_df['event_date'] > date) &
                    (events_df['event_date'] <= date + pd.Timedelta(days=h))
                ]
                
                label = 1 if len(user_events) > 0 else 0
                labels.append(label)
            
            df[f'relapse_in_{h}d'] = labels
        
        return df
    
    def prepare_tft_dataset(
        self,
        df: pd.DataFrame,
        target_horizon: int = 14,
        max_encoder_length: int = 30,
        max_prediction_length: int = 14
    ) -> TimeSeriesDataSet:
        """
        Prepara TimeSeriesDataSet para TFT
        """
        # Filter for target horizon
        df = df[df['window_size_days'] <= max_encoder_length].copy()
        
        # Add time index
        df = df.sort_values(['user_id_hash', 'date'])
        df['time_idx'] = df.groupby('user_id_hash').cumcount()
        
        # Select features
        static_categoricals = []  # No static features yet
        static_reals = []
        
        time_varying_known_categoricals = []
        time_varying_known_reals = ['window_size_days', 'num_datapoints']
        
        time_varying_unknown_categoricals = []
        time_varying_unknown_reals = [
            'sentiment_mean', 'sentiment_std', 'sentiment_trend',
            'avg_sentence_len_mean', 'ttr_mean',
            'num_messages_total', 'num_messages_mean',
            'response_latency_mean',
            'steps_mean', 'sleep_hours_mean', 'hr_mean'
        ]
        
        # Target
        target = f'relapse_in_{target_horizon}d'
        
        # Create dataset
        training = TimeSeriesDataSet(
            df,
            time_idx='time_idx',
            target=target,
            group_ids=['user_id_hash'],
            min_encoder_length=7,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=static_categoricals,
            static_reals=static_reals,
            time_varying_known_categoricals=time_varying_known_categoricals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_categoricals=time_varying_unknown_categoricals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(groups=['user_id_hash']),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        
        return training


class TFTTrainer:
    """
    Entrenador de Temporal Fusion Transformer
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.mlflow_uri = config.get('mlflow_uri', 'http://localhost:5000')
        mlflow.set_tracking_uri(self.mlflow_uri)
        
    def train(
        self,
        train_dataset: TimeSeriesDataSet,
        val_dataset: TimeSeriesDataSet,
        experiment_name: str = "ms_relapse_tft",
        **kwargs
    ) -> Tuple[TemporalFusionTransformer, Dict]:
        """
        Entrena modelo TFT
        """
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                'max_encoder_length': train_dataset.max_encoder_length,
                'max_prediction_length': train_dataset.max_prediction_length,
                'batch_size': kwargs.get('batch_size', 64),
                'learning_rate': kwargs.get('learning_rate', 0.001),
            })
            
            # Create dataloaders
            train_dataloader = train_dataset.to_dataloader(
                train=True,
                batch_size=kwargs.get('batch_size', 64),
                num_workers=0
            )
            
            val_dataloader = val_dataset.to_dataloader(
                train=False,
                batch_size=kwargs.get('batch_size', 64),
                num_workers=0
            )
            
            # Configure model
            tft = TemporalFusionTransformer.from_dataset(
                train_dataset,
                learning_rate=kwargs.get('learning_rate', 0.001),
                hidden_size=kwargs.get('hidden_size', 32),
                attention_head_size=kwargs.get('attention_head_size', 4),
                dropout=kwargs.get('dropout', 0.1),
                hidden_continuous_size=kwargs.get('hidden_continuous_size', 16),
                loss=QuantileLoss(),
                log_interval=10,
                reduce_on_plateau_patience=4,
            )
            
            # Callbacks
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=5,
                mode='min'
            )
            
            checkpoint = ModelCheckpoint(
                dirpath='./checkpoints',
                filename='tft-{epoch:02d}-{val_loss:.2f}',
                monitor='val_loss',
                mode='min'
            )
            
            # Trainer
            trainer = pl.Trainer(
                max_epochs=kwargs.get('max_epochs', 30),
                gpus=1 if torch.cuda.is_available() else 0,
                gradient_clip_val=kwargs.get('gradient_clip_val', 0.1),
                callbacks=[early_stop, checkpoint],
                enable_progress_bar=True,
            )
            
            # Train
            trainer.fit(
                tft,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader
            )
            
            # Evaluate
            metrics = self._evaluate(tft, val_dataloader, val_dataset)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.pytorch.log_model(tft, "model")
            
            logger.info(f"Training completed. Metrics: {metrics}")
            
            return tft, metrics
    
    def _evaluate(
        self,
        model: TemporalFusionTransformer,
        dataloader,
        dataset: TimeSeriesDataSet
    ) -> Dict:
        """
        Evaluación del modelo
        """
        model.eval()
        
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch in dataloader:
                pred = model(batch)
                predictions.extend(pred['prediction'].cpu().numpy())
                actuals.extend(batch['target'].cpu().numpy())
        
        predictions = np.array(predictions).flatten()
        actuals = np.array(actuals).flatten()
        
        # Binary classification metrics
        try:
            auroc = roc_auc_score(actuals, predictions)
            auprc = average_precision_score(actuals, predictions)
            brier = brier_score_loss(actuals, predictions)
        except:
            auroc, auprc, brier = 0, 0, 1
        
        metrics = {
            'auroc': auroc,
            'auprc': auprc,
            'brier_score': brier,
            'mean_prediction': float(predictions.mean()),
            'positive_rate': float(actuals.mean())
        }
        
        return metrics


# === Main Training Script ===

async def main():
    logging.basicConfig(level=logging.INFO)
    
    # Config
    config = {
        'db_dsn': 'postgresql://user:pass@localhost/empredictor',
        'mlflow_uri': 'http://localhost:5000'
    }
    
    # Load data
    logger.info("Loading data...")
    dataset_builder = MSRelapsePredictorDataset(config['db_dsn'])
    df = await dataset_builder.load_data(
        start_date='2024-01-01',
        end_date='2024-12-31'
    )
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Positive rate (14d): {df['relapse_in_14d'].mean():.3f}")
    
    # Time-series split
    unique_dates = df['date'].unique()
    split_idx = int(len(unique_dates) * 0.8)
    train_dates = unique_dates[:split_idx]
    val_dates = unique_dates[split_idx:]
    
    train_df = df[df['date'].isin(train_dates)]
    val_df = df[df['date'].isin(val_dates)]
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Prepare TFT datasets
    train_dataset = dataset_builder.prepare_tft_dataset(
        train_df,
        target_horizon=14,
        max_encoder_length=30,
        max_prediction_length=14
    )
    
    val_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset,
        val_df,
        predict=True
    )
    
    # Train
    logger.info("Starting training...")
    trainer = TFTTrainer(config)
    
    model, metrics = trainer.train(
        train_dataset,
        val_dataset,
        batch_size=32,
        learning_rate=0.001,
        max_epochs=20
    )
    
    logger.info("Training complete!")
    logger.info(f"Final metrics: {metrics}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
