#!/usr/bin/env python3
"""
GRUによるファイルアクセス予測モデル
--------------------------------

実行コマンドとアクセス履歴から、次にアクセスされるファイルを予測するGRUモデルを学習します。

特徴:
- コマンドとファイルパスの特徴量化（Embedding）
- GRUによるファイルアクセス順序の学習
- 可変長シーケンスの処理
"""

import sys
import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GRU, Dense, Dropout, Embedding, Input, TextVectorization,
    Concatenate, LayerNormalization, GlobalAveragePooling1D, Bidirectional
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from typing import Dict, List, Tuple, Set
from tqdm import tqdm

class FileAccessPredictor:
    def __init__(self, max_text_length: int = 100, max_sequence_length: int = 100):
        self.max_text_length = max_text_length
        self.max_sequence_length = max_sequence_length
        self.command_vectorizer = None
        self.filepath_encoder = None
        self.model = None
        self.df = None  # データフレームを保持
    
    def save_preprocessed_data(self, X: Dict[str, np.ndarray], y: np.ndarray, 
                             save_dir: str) -> None:
        """前処理済みデータを保存"""
        os.makedirs(save_dir, exist_ok=True)
        
        # NumPy配列を保存
        for key, value in X.items():
            np.save(os.path.join(save_dir, f'X_{key}.npy'), value)
        np.save(os.path.join(save_dir, 'y.npy'), y)
        
        # エンコーダーを保存
        with open(os.path.join(save_dir, 'command_vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.command_vectorizer, f)
        with open(os.path.join(save_dir, 'filepath_encoder.pkl'), 'wb') as f:
            pickle.dump(self.filepath_encoder, f)
        
        # データフレームを保存
        if self.df is not None:
            self.df.to_pickle(os.path.join(save_dir, 'df.pkl'))
    
    def load_preprocessed_data(self, save_dir: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """前処理済みデータを読み込み"""
        X = {}
        for key in ['command', 'history']:
            X[key] = np.load(os.path.join(save_dir, f'X_{key}.npy'))
        y = np.load(os.path.join(save_dir, 'y.npy'))
        
        # エンコーダーを読み込み
        with open(os.path.join(save_dir, 'command_vectorizer.pkl'), 'rb') as f:
            self.command_vectorizer = pickle.load(f)
        with open(os.path.join(save_dir, 'filepath_encoder.pkl'), 'rb') as f:
            self.filepath_encoder = pickle.load(f)
        
        # データフレームを読み込み
        df_path = os.path.join(save_dir, 'df.pkl')
        if os.path.exists(df_path):
            self.df = pd.read_pickle(df_path)
        
        return X, y
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """データの前処理"""
        self.df = df  # データフレームを保持
        
        print("コマンドのベクトル化...")
        if self.command_vectorizer is None:
            self.command_vectorizer = TextVectorization(
                max_tokens=10000,  # 語彙数を増やす
                output_sequence_length=self.max_text_length
            )
            self.command_vectorizer.adapt(df['command'].fillna(''))
        
        print("ファイルパスのエンコーディング...")
        if self.filepath_encoder is None:
            self.filepath_encoder = LabelEncoder()
            # 全ファイルパスを収集
            all_paths = set()
            for files_json in tqdm(df['accessed_files'].fillna('[]'), desc="ファイルパスの収集"):
                try:
                    files = json.loads(files_json)
                    all_paths.update(f['path'] for f in files)
                except json.JSONDecodeError as e:
                    print(f"JSONデコードエラー: {e}")
                    continue
            print(f"ユニークなファイルパス数: {len(all_paths)}")
            self.filepath_encoder.fit(list(all_paths))
        
        # 入力データの準備
        commands = self.command_vectorizer(df['command'].fillna(''))
        history_sequences = []
        targets = []
        
        print("シーケンスデータの生成...")
        for command_idx, files_json in tqdm(enumerate(df['accessed_files'].fillna('[]')), total=len(df)):
            try:
                files = json.loads(files_json)
                if not files:
                    continue
                
                # アクセス順序でソート
                files = sorted(files, key=lambda x: x['order'])
                
                # 各ステップでの入力と出力を作成
                for i in range(len(files)):
                    # 入力：これまでのアクセス履歴
                    history_paths = [f['path'] for f in files[:i]]
                    
                    # パディング
                    if len(history_paths) > self.max_sequence_length:
                        history_paths = history_paths[-self.max_sequence_length:]
                    while len(history_paths) < self.max_sequence_length:
                        history_paths.append('')
                    
                    # 出力：次のアクセスファイル
                    next_file = files[i]['path']
                    
                    # データを追加
                    history_encoded = [
                        self.filepath_encoder.transform([path])[0] if path else 0
                        for path in history_paths
                    ]
                    history_sequences.append(history_encoded)
                    targets.append(self.filepath_encoder.transform([next_file])[0])
            except Exception as e:
                print(f"エラー (コマンド {command_idx}): {e}")
                continue
        
        print("NumPy配列への変換...")
        X = {
            'command': commands.numpy().astype(np.int32),
            'history': np.array(history_sequences, dtype=np.int32)
        }
        y = np.array(targets, dtype=np.int32)
        
        print(f"生成されたシーケンス数: {len(y)}")
        print(f"入力シェイプ:")
        for key, value in X.items():
            print(f"- {key}: {value.shape}")
        print(f"出力シェイプ: {y.shape}")
        
        return X, y
    
    def build_model(self) -> None:
        """GRUモデルの構築"""
        # コマンド入力
        command_input = Input(shape=(self.max_text_length,), name='command', dtype=tf.int32)
        command_embedding = Embedding(10000, 128)(command_input)  # Embedding次元を増やす
        command_features = Bidirectional(GRU(128, return_sequences=True))(command_embedding)
        command_features = GlobalAveragePooling1D()(command_features)
        command_features = Dense(256, activation='relu')(command_features)
        command_features = LayerNormalization()(command_features)
        command_features = Dropout(0.5)(command_features)
        
        # アクセス履歴入力
        history_input = Input(shape=(self.max_sequence_length,), name='history', dtype=tf.int32)
        history_embedding = Embedding(
            len(self.filepath_encoder.classes_) + 1,  # +1 はパディング用
            128  # Embedding次元を増やす
        )(history_input)
        
        # GRU層
        gru = Bidirectional(GRU(256, return_sequences=True))(history_embedding)
        gru = Bidirectional(GRU(128))(gru)
        gru = LayerNormalization()(gru)
        gru = Dropout(0.5)(gru)
        
        # 特徴量の結合
        combined = Concatenate()([gru, command_features])
        
        # 出力層
        dense = Dense(512, activation='relu')(combined)
        dense = LayerNormalization()(dense)
        dense = Dropout(0.5)(dense)
        dense = Dense(256, activation='relu')(dense)
        dense = LayerNormalization()(dense)
        dense = Dropout(0.5)(dense)
        output = Dense(len(self.filepath_encoder.classes_), activation='softmax')(dense)
        
        # モデルのコンパイル
        self.model = Model(
            inputs=[command_input, history_input],
            outputs=output
        )
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),  # 学習率を下げる
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train_and_evaluate(self, X: Dict[str, np.ndarray], y: np.ndarray, 
                          n_splits: int = 5) -> List[Dict[str, float]]:
        """モデルの学習と評価（k分割交差検証）"""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X['command']), 1):
            print(f"\nFold {fold}/{n_splits}")
            
            # データの分割
            X_train = {
                'command': tf.convert_to_tensor(X['command'][train_idx], dtype=tf.int32),
                'history': tf.convert_to_tensor(X['history'][train_idx], dtype=tf.int32)
            }
            X_val = {
                'command': tf.convert_to_tensor(X['command'][val_idx], dtype=tf.int32),
                'history': tf.convert_to_tensor(X['history'][val_idx], dtype=tf.int32)
            }
            y_train = tf.convert_to_tensor(y[train_idx], dtype=tf.int32)
            y_val = tf.convert_to_tensor(y[val_idx], dtype=tf.int32)
            
            # モデルの構築
            self.build_model()
            
            # コールバック
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(  # 学習率の自動調整
                    monitor='val_loss',
                    factor=0.5,
                    patience=2,
                    min_lr=1e-6
                )
            ]
            
            # モデルの学習
            history = self.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=50,  # エポック数を増やす
                batch_size=32,  # バッチサイズを調整
                callbacks=callbacks,
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
    
    def print_example_predictions(self, X: Dict[str, np.ndarray], y: np.ndarray, 
                                n_examples: int = 5) -> None:
        """予測例の表示"""
        for i in range(min(n_examples, len(y))):
            print(f"\n予測例 {i+1}:")
            if self.df is not None:
                print(f"コマンド: {self.df['command'].iloc[i]}")
            
            # アクセス履歴の表示
            history_paths = [
                self.filepath_encoder.inverse_transform([idx])[0]
                for idx in X['history'][i]
                if idx != 0  # パディングをスキップ
            ]
            print("\nアクセス履歴:")
            for j, path in enumerate(history_paths):
                print(f"{j+1}. {path}")
            
            # 実際の次のファイル
            true_path = self.filepath_encoder.inverse_transform([y[i]])[0]
            print(f"\n実際の次のファイル: {true_path}")
            
            # 予測
            y_pred = self.model.predict({
                'command': tf.convert_to_tensor(X['command'][i:i+1], dtype=tf.int32),
                'history': tf.convert_to_tensor(X['history'][i:i+1], dtype=tf.int32)
            })[0]
            
            print("\n予測されたファイル (確率):")
            top5_indices = np.argsort(y_pred)[-5:][::-1]
            for j, idx in enumerate(top5_indices):
                path = self.filepath_encoder.inverse_transform([idx])[0]
                prob = y_pred[idx]
                print(f"{j+1}. {path}: {prob:.4f}")

def main():
    if len(sys.argv) < 2:
        print("Usage: train_gru_model.py <data_csv> [--load-data <data_dir>] [--save-data <data_dir>]")
        sys.exit(1)
    
    data_csv = sys.argv[1]
    load_data_dir = None
    save_data_dir = None
    
    # コマンドライン引数の解析
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--load-data':
            load_data_dir = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--save-data':
            save_data_dir = sys.argv[i + 1]
            i += 2
        else:
            i += 1
    
    # モデルの初期化
    predictor = FileAccessPredictor()
    
    # データの準備
    if load_data_dir:
        print(f"前処理済みデータを読み込み中: {load_data_dir}")
        X, y = predictor.load_preprocessed_data(load_data_dir)
    else:
        # データの読み込みと前処理
        print("データの読み込み中...")
        df = pd.read_csv(data_csv)
        print(f"読み込んだデータ: {len(df)}行")
        
        print("データの前処理中...")
        X, y = predictor.preprocess_data(df)
        
        # 前処理済みデータの保存
        if save_data_dir:
            print(f"前処理済みデータを保存中: {save_data_dir}")
            predictor.save_preprocessed_data(X, y, save_data_dir)
    
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
    predictor.print_example_predictions(X, y)

if __name__ == "__main__":
    main()
