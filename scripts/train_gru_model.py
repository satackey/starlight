#!/usr/bin/env python3
"""
GRUによるファイル予測モデル
--------------------------

convert_traces_to_csv.py で生成された学習データを使用して、
次にアクセスされるファイルを予測するGRUモデルを学習します。

特徴:
- 10分割交差検証による評価
- 複数の特徴量を活用（ファイルパス、サイズ、タイプなど）
- Early Stoppingによる過学習防止
"""

import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Embedding, Input
from tensorflow.keras.callbacks import EarlyStopping
from typing import Dict, List, Tuple

class FileAccessPredictor:
    def __init__(self, sequence_length: int = 5):
        self.sequence_length = sequence_length
        self.label_encoders = {}
        self.scalers = {}
        self.model = None
        self.vocab_sizes = {}
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """データの前処理"""
        # カテゴリカル変数のエンコーディング
        categorical_features = ['input_files', 'input_types', 'target_file', 'target_type']
        for feature in categorical_features:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
            
            if feature.startswith('input_'):
                # シーケンスデータの処理
                sequences = df[feature].apply(json.loads)
                flat_values = [item for seq in sequences for item in seq]
                self.label_encoders[feature].fit(flat_values)
                encoded_sequences = sequences.apply(
                    lambda x: self.label_encoders[feature].transform(x)
                )
                df[f'{feature}_encoded'] = encoded_sequences
                self.vocab_sizes[feature] = len(self.label_encoders[feature].classes_)
            else:
                # ターゲットデータの処理
                self.label_encoders[feature].fit(df[feature])
                df[f'{feature}_encoded'] = self.label_encoders[feature].transform(df[feature])
                self.vocab_sizes[feature] = len(self.label_encoders[feature].classes_)
        
        # 数値変数のスケーリング
        numerical_features = ['input_sizes', 'input_modes', 'target_size', 'target_mode']
        for feature in numerical_features:
            if feature not in self.scalers:
                self.scalers[feature] = StandardScaler()
            
            if feature.startswith('input_'):
                # シーケンスデータの処理
                sequences = df[feature].apply(json.loads)
                flat_values = np.array([item for seq in sequences for item in seq]).reshape(-1, 1)
                self.scalers[feature].fit(flat_values)
                encoded_sequences = sequences.apply(
                    lambda x: self.scalers[feature].transform(np.array(x).reshape(-1, 1)).flatten()
                )
                df[f'{feature}_scaled'] = encoded_sequences
            else:
                # ターゲットデータの処理
                self.scalers[feature].fit(df[feature].values.reshape(-1, 1))
                df[f'{feature}_scaled'] = self.scalers[feature].transform(
                    df[feature].values.reshape(-1, 1)
                ).flatten()
        
        # 特徴量とターゲットの準備
        X = {
            'file_sequence': np.array(df['input_files_encoded'].tolist()),
            'type_sequence': np.array(df['input_types_encoded'].tolist()),
            'size_sequence': np.array(df['input_sizes_scaled'].tolist()),
            'mode_sequence': np.array(df['input_modes_scaled'].tolist())
        }
        y = df['target_file_encoded'].values
        
        return X, y
    
    def build_model(self) -> None:
        """GRUモデルの構築"""
        # 入力レイヤー
        file_input = Input(shape=(self.sequence_length,), name='file_sequence')
        type_input = Input(shape=(self.sequence_length,), name='type_sequence')
        size_input = Input(shape=(self.sequence_length,), name='size_sequence')
        mode_input = Input(shape=(self.sequence_length,), name='mode_sequence')
        
        # Embedding層
        file_embedding = Embedding(
            self.vocab_sizes['input_files'], 32
        )(file_input)
        type_embedding = Embedding(
            self.vocab_sizes['input_types'], 16
        )(type_input)
        
        # 数値特徴量の reshape
        size_reshape = tf.keras.layers.Reshape(
            (self.sequence_length, 1)
        )(size_input)
        mode_reshape = tf.keras.layers.Reshape(
            (self.sequence_length, 1)
        )(mode_input)
        
        # 特徴量の結合
        merged = tf.keras.layers.Concatenate()(
            [file_embedding, type_embedding, size_reshape, mode_reshape]
        )
        
        # GRU層
        gru1 = GRU(128, return_sequences=True)(merged)
        dropout1 = Dropout(0.3)(gru1)
        gru2 = GRU(64)(dropout1)
        dropout2 = Dropout(0.3)(gru2)
        
        # 出力層
        dense = Dense(32, activation='relu')(dropout2)
        output = Dense(self.vocab_sizes['target_file'], activation='softmax')(dense)
        
        # モデルのコンパイル
        self.model = tf.keras.Model(
            inputs=[file_input, type_input, size_input, mode_input],
            outputs=output
        )
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train_and_evaluate(self, X: Dict[str, np.ndarray], y: np.ndarray, 
                          n_splits: int = 10) -> List[Dict[str, float]]:
        """モデルの学習と評価（k分割交差検証）"""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X['file_sequence']), 1):
            print(f"\nFold {fold}/{n_splits}")
            
            # データの分割
            X_train = {k: v[train_idx] for k, v in X.items()}
            X_val = {k: v[val_idx] for k, v in X.items()}
            y_train = y[train_idx]
            y_val = y[val_idx]
            
            # モデルの構築
            self.build_model()
            
            # Early Stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            # モデルの学習
            history = self.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=1
            )
            
            # 評価
            val_loss, val_acc = self.model.evaluate(X_val, y_val, verbose=0)
            
            # Top-5精度の計算
            y_pred = self.model.predict(X_val)
            top5_acc = self.calculate_top_k_accuracy(y_val, y_pred, k=5)
            
            fold_results.append({
                'fold': fold,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'top5_accuracy': top5_acc
            })
            
            # モデルの保存
            self.model.save(f'file_prediction_model_fold{fold}.h5')
        
        return fold_results
    
    def calculate_top_k_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               k: int = 5) -> float:
        """Top-k精度の計算"""
        top_k_pred = np.argsort(y_pred, axis=1)[:, -k:]
        correct = 0
        for i, true_label in enumerate(y_true):
            if true_label in top_k_pred[i]:
                correct += 1
        return correct / len(y_true)
    
    def print_example_predictions(self, X: Dict[str, np.ndarray], y: np.ndarray, 
                                n_examples: int = 5) -> None:
        """予測例の表示"""
        y_pred = self.model.predict(X)
        top5_pred = np.argsort(y_pred, axis=1)[:, -5:]
        
        for i in range(min(n_examples, len(y))):
            print(f"\n予測例 {i+1}:")
            true_file = self.label_encoders['target_file'].inverse_transform([y[i]])[0]
            print(f"実際のファイル: {true_file}")
            
            print("Top-5予測:")
            for pred_idx in reversed(top5_pred[i]):
                pred_file = self.label_encoders['target_file'].inverse_transform([pred_idx])[0]
                prob = y_pred[i][pred_idx]
                print(f"- {pred_file}: {prob:.4f}")

def main():
    if len(sys.argv) != 2:
        print("Usage: train_gru_model.py <data_csv>")
        sys.exit(1)
    
    data_csv = sys.argv[1]
    
    # データの読み込み
    print("データの読み込み中...")
    df = pd.read_csv(data_csv)
    
    # モデルの初期化と学習
    predictor = FileAccessPredictor()
    
    print("データの前処理中...")
    X, y = predictor.preprocess_data(df)
    
    print("モデルの学習と評価中...")
    results = predictor.train_and_evaluate(X, y)
    
    # 結果の表示
    print("\n評価結果:")
    for fold_result in results:
        print(f"\nFold {fold_result['fold']}:")
        print(f"- 検証損失: {fold_result['val_loss']:.4f}")
        print(f"- 検証精度: {fold_result['val_accuracy']:.4f}")
        print(f"- Top-5精度: {fold_result['top5_accuracy']:.4f}")
    
    # 平均スコアの計算
    avg_acc = np.mean([r['val_accuracy'] for r in results])
    avg_top5 = np.mean([r['top5_accuracy'] for r in results])
    print(f"\n平均スコア:")
    print(f"- 精度: {avg_acc:.4f}")
    print(f"- Top-5精度: {avg_top5:.4f}")
    
    # 予測例の表示
    print("\n予測例:")
    predictor.print_example_predictions(X, y)

if __name__ == "__main__":
    main()
