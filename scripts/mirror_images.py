"""
Image Mirroring Tool

このスクリプトは、CSVファイルに記載されたDockerイメージを
プライベートレジストリにミラーリングするためのツールです。

目的:
    - CSVファイルから一意なベースイメージのリストを抽出
    - 各イメージをプライベートレジストリ(cloud.cluster.local:5000)にミラーリング
    - ミラーリングの進捗と結果を表示

使用方法:
    python mirror_images.py <path_to_csv_file>

引数:
    <path_to_csv_file>: Base Imageカラムを含むCSVファイルへのパス

必要条件:
    - containerdのctrコマンドがインストールされていること
    - プライベートレジストリ(cloud.cluster.local:5000)にアクセス可能であること
    - CSVファイルが'Base Image'カラムを含んでいること

出力:
    - 各イメージのpull/tag/push操作の進捗
    - 成功/失敗したイメージの数
"""

import csv
import subprocess
from pathlib import Path
import argparse

def read_base_images(csv_path):
    base_images = set()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            base_images.add(row['Base Image'])
    return list(base_images)

def mirror_image(source_tag):
    
    mirror_tag = f"cloud.cluster.local:5000/mirror/{source_tag.split('/')[-1]}"

    # Add docker.io prefix if no registry is specified
    if '/' not in source_tag or not any(x in source_tag.split('/')[0] for x in ['.', ':']):
        source_tag = f"docker.io/library/{source_tag}"
    
    try:
        # check if image already exists in local
        print(f"Checking if {source_tag} is already present...")
        try:
            subprocess.run(f"sudo ctr images ls | grep {mirror_tag}", check=True, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"{source_tag} already exists locally")
        except:
            # Pull original image
            print(f"Pulling {source_tag}...")
            subprocess.run(['sudo', 'ctr', 'images', 'pull', source_tag], check=True)
            
            # Tag for mirror
            print(f"Tagging {source_tag} as {mirror_tag}...")
            subprocess.run(['sudo', 'ctr', 'images', 'tag', source_tag, mirror_tag], check=True)
            
        # Push to mirror
        print(f"Pushing to {mirror_tag}...")
        subprocess.run(['sudo', 'ctr', 'images', 'push', '--platform', 'linux/amd64', '--plain-http', mirror_tag], check=True)
        
        print(f"Successfully mirrored {source_tag}")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Error mirroring {source_tag}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Mirror Docker images from CSV file')
    parser.add_argument('csv_file', type=str, help='Path to CSV file containing base images')
    args = parser.parse_args()

    # Validate CSV file path
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return 1

    # Get unique base images
    base_images = read_base_images(csv_path)
    print(f"Found {len(base_images)} unique base images")
    
    # Mirror each image
    success_count = 0
    for image in base_images:
        if mirror_image(image):
            success_count += 1
    
    print(f"\nMirroring complete: {success_count}/{len(base_images)} images processed successfully")
    return 0

if __name__ == "__main__":
    exit(main())
