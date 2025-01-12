#!/usr/bin/env python3
"""
GRUによるファイルアクセス予測モデル
--------------------------------

実行コマンドとアクセス履歴から、次にアクセスされるファイルを予測するGRUモデルを学習します。

特徴:
- コマンドとファイルパスの特徴量化（Embedding）
- GRUによるファイルアクセス順序の学習
- Attentionによるアクセス履歴の活用
- 可変長シーケンスの処理
"""

import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GRU, Dense, Dropout, Embedding, Input, TextVectorization,
    Concatenate, Attention, LayerNormalization, Masking
)
from tensorflow.keras.callbacks import EarlyStopping
from typing import Dict, List, Tuple, Set

class FileAccessPredictor:
    def __init__(self, max_text_length: int = 100, max_sequence_length: int = 100):
        self.max_text_length = max_text_length
        self.max_sequence_length = max_sequence_length
        self.command_vectorizer = None
        self.filepath_encoder = None
        self.model = None
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """データの前処理"""
        # コマンドのベクトル化
        if self.command_vectorizer is None:
            self.command_vectorizer = TextVectorization(
                max_tokens=5000,
                output_sequence_length=self.max_text_length
            )
            self.command_vectorizer.adapt(df['command'])
        
        # ファイルパスのエンコーディング
        if self.filepath_encoder is None:
            self.filepath_encoder = LabelEncoder()
            # 全ファイルパスを収集
            all_paths = set()
            for files_json in df['accessed_files']:
                files = json.loads(files_json)
                all_paths.update(f['path'] for f in files)
            self.filepath_encoder.fit(list(all_paths))
        
        # 入力データの準備
        X = {
            'command': self.command_vectorizer(df['command']),
            'history': [],
            'history_sizes': []
        }
        y = []
        
        # 各コマンドのアクセス履歴を処理
        for command_idx, files_json in enumerate(df['accessed_files']):
            files = json.loads(files_json)
            # アクセス順序でソート
            files = sorted(files, key=lambda x: x['order'])
            
            # 各ステップでの入力と出力を作成
            for i in range(len(files)):
                # 入力：これまでのアクセス履歴
                history_paths = [f['path'] for f in files[:i]]
                history_sizes = [f['size'] for f in files[:i]]
                
                # パディング
                while len(history_paths) < self.max_sequence_length:
                    history_paths.append('')  # パディング用の特別な値
                    history_sizes.append(0)
                
                # 出力：次のアクセスファイル
                next_file = files[i]['path']
                
                # データを追加
                X['history'].append([
                    self.filepath_encoder.transform([path])[0] if path else -1
                    for path in history_paths[:self.max_sequence_length]
                ])
                X['history_sizes'].append(history_sizes[:self.max_sequence_length])
                y.append(self.filepath_encoder.transform([next_file])[0])
        
        # NumPy配列に変換
        X['history'] = np.array(X['history'])
        X['history_sizes'] = np.array(X['history_sizes'])
        y = np.array(y)
        
        return X, y
    
    def build_model(self) -> None:
        """GRUモデルの構築"""
        # コマンド入力
        command_input = Input(shape=(self.max_text_length,), name='command')
        command_embedding = Embedding(5000, 64)(command_input)
        command_features = Dense(128)(command_embedding)
        
        # アクセス履歴入力
        history_input = Input(shape=(self.max_sequence_length,), name='history')
        history_embedding = Embedding(
            len(self.filepath_encoder.classes_) + 1,  # +1 はパディング用
            32,
            mask_zero=True
        )(history_input)
        
        # サイズ入力
        sizes_input = Input(shape=(self.max_sequence_length,), name='history_sizes')
        sizes_reshape = tf.keras.layers.Reshape((self.max_sequence_length, 1))(sizes_input)
        
        # 特徴量の結合
        features = Concatenate(axis=-1)([history_embedding, sizes_reshape])
        
        # GRU層（アクセス順序の学習）
        gru = GRU(256, return_sequences=True)(features)
        gru = LayerNormalization()(gru)
        gru = Dropout(0.3)(gru)
        
        # Attention層（アクセス履歴との関連性）
        attention = Attention()([gru, command_features])
        
        # 出力層（次のファイルの予測）
        dense = Dense(256, activation='relu')(attention)
        dropout = Dropout(0.3)(dense)
        output = Dense(len(self.filepath_encoder.classes_), activation='softmax')(dropout)
        
        # モデルのコンパイル
        self.model = Model(
            inputs=[command_input, history_input, sizes_input],
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
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X['command']), 1):
            print(f"\nFold {fold}/{n_splits}")
            
            # データの分割
            X_train = {
                'command': tf.gather(X['command'], train_idx),
                'history': tf.gather(X['history'], train_idx),
                'history_sizes': tf.gather(X['history_sizes'], train_idx)
            }
            X_val = {
                'command': tf.gather(X['command'], val_idx),
                'history': tf.gather(X['history'], val_idx),
                'history_sizes': tf.gather(X['history_sizes'], val_idx)
            }
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
            
            # Top-k精度の計算
            y_pred = self.model.predict(X_val)
            top5_acc = self.calculate_top_k_accuracy(y_val, y_pred, k=5)
            
            fold_results.append({
                'fold': fold,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'top5_accuracy': top5_acc
            })
            
            # モデルの保存
            self.model.save(f'file_access_prediction_model_fold{fold}.h5')
        
        return fold_results
    
    def calculate_top_k_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, k: int = 5) -> float:
        """Top-k精度の計算"""
        top_k_pred = np.argsort(y_pred, axis=1)[:, -k:]
        correct = 0
        total = 0
        
        for i in range(len(y_true)):
            if y_true[i] in top_k_pred[i]:
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0
    
    def print_example_predictions(self, df: pd.DataFrame, X: Dict[str, np.ndarray], y: np.ndarray, 
                                n_examples: int = 5) -> None:
        """予測例の表示"""
        for i in range(min(n_examples, len(y))):
            print(f"\n予測例 {i+1}:")
            print(f"コマンド: {df['command'].iloc[i]}")
            
            # アクセス履歴の表示
            history_paths = [
                self.filepath_encoder.inverse_transform([idx])[0]
                for idx in X['history'][i]
                if idx != -1  # パディングをスキップ
            ]
            print("\nアクセス履歴:")
            for j, path in enumerate(history_paths):
                size = X['history_sizes'][i][j]
                print(f"{j+1}. {path} (size: {size})")
            
            # 実際の次のファイル
            true_path = self.filepath_encoder.inverse_transform([y[i]])[0]
            print(f"\n実際の次のファイル: {true_path}")
            
            # 予測
            y_pred = self.model.predict({
                'command': X['command'][i:i+1],
                'history': X['history'][i:i+1],
                'history_sizes': X['history_sizes'][i:i+1]
            })[0]
            
            print("\n予測されたファイル (確率):")
            top5_indices = np.argsort(y_pred)[-5:][::-1]
            for j, idx in enumerate(top5_indices):
                path = self.filepath_encoder.inverse_transform([idx])[0]
                prob = y_pred[idx]
                print(f"{j+1}. {path}: {prob:.4f}")

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
    print(f"\n全体の平均スコア:")
    print(f"- 精度: {avg_acc:.4f}")
    print(f"- Top-5精度: {avg_top5:.4f}")
    
    # 予測例の表示
    print("\n予測例:")
    predictor.print_example_predictions(df, X, y)

if __name__ == "__main__":
    main()
