"""
Fine-tuning por Paciente - EM Predictor
Entrena un modelo personalizado para cada paciente usando transfer learning.

Estrategia:
1. Modelo base: Entrenado con datos sintéticos o agregados (si disponibles)
2. Fine-tune: Ajuste fino con datos específicos del paciente
3. Personalización: Aprende el "baseline" individual de cada paciente

Responsable: Agent ML (Brain)

Uso:
    python -m scripts.etl.finetune_patient \
        --patient-data data/processed/P001/training_dataset.parquet \
        --base-model models/tft_base.pt \
        --output models/patients/P001/
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finetune-patient")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class FinetuneConfig:
    """Configuración de fine-tuning."""
    
    # Datos
    patient_data_path: Path
    base_model_path: Optional[Path] = None
    output_path: Path = Path("models/patients")
    
    # Entrenamiento
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-4  # Más bajo que entrenamiento inicial
    weight_decay: float = 0.01
    
    # Fine-tuning específico
    freeze_embeddings: bool = True  # Congelar capas de embedding
    freeze_layers: int = 0  # Número de capas transformer a congelar
    
    # Regularización
    dropout: float = 0.1
    label_smoothing: float = 0.1
    
    # Early stopping
    patience: int = 3
    min_delta: float = 0.001
    
    # Horizonte objetivo
    target_horizon: int = 14
    
    # Features
    feature_columns: List[str] = field(default_factory=lambda: [
        "messages_mean", "messages_std", "messages_trend",
        "words_mean", "ttr_mean",
        "sentiment_mean", "sentiment_std", "sentiment_trend",
        "active_hours_mean", "night_ratio_mean",
        "coverage",
    ])


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════


class PatientDataset(Dataset):
    """Dataset para un paciente específico."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        sequence_length: int = 30,
        stride: int = 1,
    ):
        self.df = df.sort_values("date").reset_index(drop=True)
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.sequence_length = sequence_length
        self.stride = stride
        
        # Filtrar solo columnas disponibles
        available_cols = [c for c in feature_columns if c in df.columns]
        if len(available_cols) < len(feature_columns):
            missing = set(feature_columns) - set(available_cols)
            logger.warning(f"Columnas no encontradas: {missing}")
        self.feature_columns = available_cols
        
        # Normalizar features (z-score por paciente)
        self.feature_means = df[self.feature_columns].mean()
        self.feature_stds = df[self.feature_columns].std().replace(0, 1)
        
        # Crear índices de secuencias válidas
        self.valid_indices = self._compute_valid_indices()
    
    def _compute_valid_indices(self) -> List[int]:
        """Calcula índices donde podemos formar secuencias completas."""
        indices = []
        for i in range(0, len(self.df) - self.sequence_length, self.stride):
            # Verificar que la secuencia tiene target válido
            end_idx = i + self.sequence_length - 1
            if pd.notna(self.df.iloc[end_idx][self.target_column]):
                indices.append(i)
        return indices
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.sequence_length
        
        # Features
        features = self.df.iloc[start_idx:end_idx][self.feature_columns].values
        features = (features - self.feature_means.values) / self.feature_stds.values
        features = np.nan_to_num(features, nan=0.0)
        
        # Target (del último timestep)
        target = self.df.iloc[end_idx - 1][self.target_column]
        
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )
    
    def get_normalization_params(self) -> Dict[str, Dict[str, float]]:
        """Retorna parámetros de normalización para inferencia."""
        return {
            "means": self.feature_means.to_dict(),
            "stds": self.feature_stds.to_dict(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# MODELO SIMPLIFICADO PARA FINE-TUNING
# ══════════════════════════════════════════════════════════════════════════════


class PatientRiskModel(nn.Module):
    """
    Modelo de riesgo personalizado por paciente.
    
    Arquitectura simplificada:
    - LSTM bidireccional para capturar patrones temporales
    - Attention para ponderar timesteps importantes
    - Head de clasificación binaria
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Proyección de entrada
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # LSTM bidireccional
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Clasificador
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        
        Returns:
            (batch,) probabilidades de riesgo
        """
        # Proyección
        x = self.input_proj(x)  # (batch, seq_len, hidden_dim)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim * 2)
        
        # Attention weights
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Weighted sum
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden_dim * 2)
        
        # Clasificación
        output = self.classifier(context).squeeze(-1)  # (batch,)
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna pesos de atención para interpretabilidad."""
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        attn_weights = self.attention(lstm_out)
        return torch.softmax(attn_weights, dim=1).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINER
# ══════════════════════════════════════════════════════════════════════════════


class PatientFinetuner:
    """Entrena/fine-tunea modelo para un paciente específico."""
    
    def __init__(self, config: FinetuneConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Usando dispositivo: {self.device}")
    
    def load_data(self) -> Tuple[PatientDataset, PatientDataset]:
        """Carga y divide datos del paciente."""
        df = pd.read_parquet(self.config.patient_data_path)
        
        # Filtrar por ventana de 14 días (o la configurada)
        target_col = f"relapse_in_{self.config.target_horizon}d"
        if target_col not in df.columns:
            raise ValueError(f"Columna objetivo no encontrada: {target_col}")
        
        # Filtrar solo ventanas de 14 días
        df = df[df["window_size_days"] == 14].copy()
        
        if len(df) < 30:
            raise ValueError(f"Datos insuficientes: {len(df)} registros (mínimo 30)")
        
        # Split temporal (80/20)
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]
        
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")
        logger.info(f"Positive rate train: {train_df[target_col].mean():.2%}")
        logger.info(f"Positive rate val: {val_df[target_col].mean():.2%}")
        
        train_dataset = PatientDataset(
            train_df,
            self.config.feature_columns,
            target_col,
            sequence_length=30,
        )
        
        val_dataset = PatientDataset(
            val_df,
            self.config.feature_columns,
            target_col,
            sequence_length=30,
        )
        
        return train_dataset, val_dataset
    
    def create_model(self, input_dim: int) -> PatientRiskModel:
        """Crea o carga modelo base."""
        model = PatientRiskModel(
            input_dim=input_dim,
            hidden_dim=64,
            num_layers=2,
            dropout=self.config.dropout,
        )
        
        # Cargar pesos base si existen
        if self.config.base_model_path and self.config.base_model_path.exists():
            logger.info(f"Cargando modelo base: {self.config.base_model_path}")
            checkpoint = torch.load(
                self.config.base_model_path,
                map_location=self.device,
                weights_only=False,
            )
            
            # Cargar solo pesos compatibles
            model_dict = model.state_dict()
            pretrained_dict = {
                k: v for k, v in checkpoint["model_state_dict"].items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            logger.info(f"Cargados {len(pretrained_dict)}/{len(model_dict)} pesos")
        
        return model.to(self.device)
    
    def train(
        self,
        model: PatientRiskModel,
        train_dataset: PatientDataset,
        val_dataset: PatientDataset,
    ) -> Dict[str, List[float]]:
        """Entrena el modelo."""
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        # Optimizador con weight decay
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Loss con label smoothing
        criterion = nn.BCELoss()
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
        
        history = {"train_loss": [], "val_loss": [], "val_auc": []}
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            # Training
            model.train()
            train_losses = []
            
            for features, targets in train_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(features)
                
                # Label smoothing
                if self.config.label_smoothing > 0:
                    targets = targets * (1 - self.config.label_smoothing) + 0.5 * self.config.label_smoothing
                
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                train_losses.append(loss.item())
            
            # Validation
            model.eval()
            val_losses = []
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for features, targets in val_loader:
                    features = features.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                    
                    val_losses.append(loss.item())
                    val_preds.extend(outputs.cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())
            
            # Métricas
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            
            # AUC (si hay varianza en targets)
            try:
                from sklearn.metrics import roc_auc_score
                val_auc = roc_auc_score(val_targets, val_preds)
            except:
                val_auc = 0.5
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_auc"].append(val_auc)
            
            logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val AUC: {val_auc:.4f}"
            )
            
            # Scheduler step
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss - self.config.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                # Guardar mejor modelo
                self._save_checkpoint(model, train_dataset, history, is_best=True)
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping en epoch {epoch+1}")
                    break
        
        return history
    
    def _save_checkpoint(
        self,
        model: PatientRiskModel,
        dataset: PatientDataset,
        history: Dict[str, List[float]],
        is_best: bool = False,
    ):
        """Guarda checkpoint del modelo."""
        output_dir = self.config.output_path
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "input_dim": model.input_dim,
                "hidden_dim": model.hidden_dim,
            },
            "normalization": dataset.get_normalization_params(),
            "feature_columns": dataset.feature_columns,
            "target_horizon": self.config.target_horizon,
            "history": history,
            "timestamp": datetime.now().isoformat(),
        }
        
        filename = "best_model.pt" if is_best else "last_model.pt"
        torch.save(checkpoint, output_dir / filename)
        
        # Guardar config
        config_path = output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump({
                "feature_columns": dataset.feature_columns,
                "target_horizon": self.config.target_horizon,
                "epochs_trained": len(history["train_loss"]),
                "best_val_loss": min(history["val_loss"]),
                "best_val_auc": max(history["val_auc"]),
            }, f, indent=2)
    
    def run(self) -> Dict[str, any]:
        """Ejecuta el pipeline de fine-tuning."""
        logger.info("=== FINE-TUNING POR PACIENTE ===")
        
        # Cargar datos
        train_dataset, val_dataset = self.load_data()
        
        # Crear modelo
        input_dim = len(train_dataset.feature_columns)
        model = self.create_model(input_dim)
        logger.info(f"Modelo creado con {sum(p.numel() for p in model.parameters())} parámetros")
        
        # Entrenar
        history = self.train(model, train_dataset, val_dataset)
        
        # Resultado
        result = {
            "status": "success",
            "epochs_trained": len(history["train_loss"]),
            "best_val_loss": min(history["val_loss"]),
            "best_val_auc": max(history["val_auc"]),
            "output_path": str(self.config.output_path),
        }
        
        logger.info(f"Fine-tuning completado: {result}")
        return result


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tuning de modelo por paciente"
    )
    parser.add_argument(
        "--patient-data", "-d",
        type=Path,
        required=True,
        help="Archivo parquet con datos del paciente",
    )
    parser.add_argument(
        "--base-model", "-b",
        type=Path,
        default=None,
        help="Modelo base para transfer learning",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("models/patients/default"),
        help="Directorio de salida",
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=10,
        help="Número de epochs",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=14,
        help="Horizonte de predicción (días)",
    )
    
    args = parser.parse_args()
    
    config = FinetuneConfig(
        patient_data_path=args.patient_data,
        base_model_path=args.base_model,
        output_path=args.output,
        epochs=args.epochs,
        target_horizon=args.horizon,
    )
    
    finetuner = PatientFinetuner(config)
    result = finetuner.run()
    
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

