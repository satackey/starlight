import subprocess
import json
from typing import List, Dict, Tuple, Optional, Set
import re
from collections import defaultdict
import sys
import time
from pathlib import Path
import urllib.request
import urllib.error
import ssl
import os
from datetime import datetime, timedelta

class DockerHubCache:
    CACHE_FILE = "convert_images.cache.json"
    CACHE_TTL = timedelta(hours=24)

    def __init__(self):
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        if os.path.exists(self.CACHE_FILE):
            try:
                with open(self.CACHE_FILE, 'r') as f:
                    data = json.load(f)
                    now = datetime.now()
                    data = {
                        k: v for k, v in data.items()
                        if datetime.fromisoformat(v['timestamp']) + self.CACHE_TTL > now
                    }
                    return data
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Cache file corrupted, creating new cache: {e}")
                return {}
        return {}

    def _save_cache(self):
        with open(self.CACHE_FILE, 'w') as f:
            json.dump(self.cache, f, indent=2)

    def get(self, image: str) -> Optional[List[str]]:
        if image in self.cache:
            entry = self.cache[image]
            cache_time = datetime.fromisoformat(entry['timestamp'])
            if cache_time + self.CACHE_TTL > datetime.now():
                return entry['tags']
        return None

    def set(self, image: str, tags: List[str]):
        self.cache[image] = {
            'tags': tags,
            'timestamp': datetime.now().isoformat()
        }
        self._save_cache()

class DockerTagOptimizer:
    def __init__(self, cloud_host: str):
        self.cloud_host = cloud_host
        self.cached_registry_tags: Dict[str, Set[str]] = {}
        self.docker_cache = DockerHubCache()

    def fetch_docker_tags(self, image: str) -> List[str]:
        cached_tags = self.docker_cache.get(image)
        if cached_tags is not None:
            print(f"  Using cached tags for {image}")
            return cached_tags

        try:
            cmd = f"skopeo list-tags docker://docker.io/library/{image}"
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            tags_data = json.loads(result.stdout)
            tags = tags_data['Tags']
            
            self.docker_cache.set(image, tags)
            return tags
        except Exception as e:
            print(f"Error fetching tags for {image}: {str(e)}")
            return []

    def fetch_registry_tags(self, image: str) -> Set[str]:
        if image in self.cached_registry_tags:
            return self.cached_registry_tags[image]

        try:
            url = f"http://{self.cloud_host}/v2/{image}/tags/list"
            response = urllib.request.urlopen(url)
            data = json.loads(response.read().decode())
            tags = set(data.get('tags', []))
            self.cached_registry_tags[image] = tags
            return tags
        except (urllib.error.URLError, json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ Failed to fetch registry tags for {image}: {str(e)}")
            self.cached_registry_tags[image] = set()
            return set()

    def check_registry_tag(self, image: str, tag: str) -> bool:
        if image not in self.cached_registry_tags:
            self.fetch_registry_tags(image)
        return tag in self.cached_registry_tags[image]

    def parse_version(self, tag: str) -> Tuple[List[int], str, str]:
        version_match = re.search(r'(\d+(?:\.\d+)*)', tag)
        if not version_match:
            return ([], '', '')
        
        version_str = version_match.group(1)
        version_numbers = [int(x) for x in version_str.split('.')]
        start = version_match.start()
        end = version_match.end()
        
        prefix = tag[:start].rstrip('-')
        suffix = tag[end:].lstrip('-')
        
        return (version_numbers, prefix, suffix)

    def group_tags_by_version_depth(self, tags: List[str]) -> Dict[int, List[str]]:
        depth_groups = defaultdict(list)
        for tag in tags:
            version_numbers, _, _ = self.parse_version(tag)
            if version_numbers:
                depth = len(version_numbers)
                depth_groups[depth].append(tag)
        return depth_groups

    def find_consistent_tags(self, tags: List[str], preferred_depth: int = 3) -> List[str]:
        depth_groups = self.group_tags_by_version_depth(tags)
        if not depth_groups:
            return []

        depths = sorted(depth_groups.keys(), reverse=True)
        
        for depth in depths:
            depth_tags = depth_groups[depth]
            variant_groups = defaultdict(list)
            
            for tag in depth_tags:
                version_nums, prefix, suffix = self.parse_version(tag)
                if not version_nums:
                    continue
                    
                variant_key = f"{prefix}|{suffix}"
                variant_groups[variant_key].append((version_nums, tag))
            
            for variant_tags in variant_groups.values():
                if len(variant_tags) >= 3:
                    sorted_tags = sorted(variant_tags, key=lambda x: x[0], reverse=True)
                    return [tag for _, tag in sorted_tags[:3]]
        
        return []

    def filter_unsuitable_tags(self, tags: List[str]) -> List[str]:
        exclude_patterns = [
            r'alpha', r'beta', r'rc\d*',
            r'windowsservercore', r'nanoserver',
            r'preview', r'test', r'-?dev',
            r'latest'
        ]
        
        return [tag for tag in tags if not any(
            re.search(pattern, tag, re.IGNORECASE) 
            for pattern in exclude_patterns
        )]

    def cleanup_container(self, container_name: str):
        self.run_command(f"sudo ctr task rm -f {container_name}", 
                        f"Removing task {container_name}",
                        ignore_error=True)
        
        self.run_command(f"sudo ctr container rm {container_name}",
                        f"Removing container {container_name}",
                        ignore_error=True)

    def run_command(self, cmd: str, description: str = "", ignore_error: bool = False) -> bool:
        print(f"\n=== {description} ===")
        print(f"Executing: {cmd}")
        
        try:
            result = subprocess.run(cmd, shell=True, text=True, 
                                  capture_output=True)
            
            if result.stdout.strip():
                print("Output:")
                print(result.stdout)
            
            if result.returncode == 0:
                print(f"✅ {description} completed successfully")
                return True
            else:
                if ignore_error:
                    print(f"⚠️ {description} failed (ignored)")
                    return True
                else:
                    print(f"❌ Error during {description}:")
                    print(result.stderr)
                    return False
                
        except Exception as e:
            if ignore_error:
                print(f"⚠️ Failed to execute {description} (ignored):")
                print(str(e))
                return True
            else:
                print(f"❌ Failed to execute {description}:")
                print(str(e))
                return False

    def should_convert_image(self, image: str, tag: str) -> bool:
        starlight_tag = f"{tag}-starlight"
        return not self.check_registry_tag(image, starlight_tag)

    def convert_image(self, image: str, tag: str) -> bool:
        """単一のイメージを変換"""
        starlight_tag = f"{tag}-starlight"
        source = f"docker.io/library/{image}:{tag}"
        destination = f"{self.cloud_host}/{image}:{starlight_tag}"
        cmd = f"""sudo ctr-starlight convert \
            --insecure-destination \
            --notify --profile myproxy \
            --platform linux/amd64 \
            {source} {destination}"""
        
        return self.run_command(cmd, f"Converting {source} to Starlight format")

    def optimize_and_report(self, image: str, tags: List[str]) -> bool:
        """イメージの最適化とレポート生成を実行"""
        success = True
        starlight_tags = [f"{tag}-starlight" for tag in tags]

        if not self.run_command("sudo ctr-starlight optimizer on", 
                              "Starting optimizer"):
            return False

        test_dir = "/tmp/test-data"
        Path(test_dir).mkdir(parents=True, exist_ok=True)
        
        for tag in starlight_tags:
            destination = f"{self.cloud_host}/{image}:{tag}"
            container_name = f"test-{image}-{tag}"
            
            try:
                self.cleanup_container(container_name)

                if not self.run_command(
                    f"sudo ctr-starlight pull --profile myproxy {destination}",
                    f"Pulling {destination}"
                ):
                    success = False
                    break

                if not self.run_command(
                    f"""sudo ctr c create \
                        --snapshotter=starlight \
                        --mount type=bind,src={test_dir},dst=/data,options=rbind:rw \
                        {destination} \
                        {container_name} /bin/sh""",
                    f"Creating container {container_name}"
                ):
                    success = False
                    break

                if not self.run_command(
                    f"sudo ctr task start {container_name}",
                    f"Starting container {container_name}"
                ):
                    success = False
                    break

                print("\nRunning container for 30 seconds:")
                for i in range(30):
                    print(f"\r[{'=' * i}{' ' * (29-i)}] {i+1}/30s", end='', flush=True)
                    time.sleep(1)
                print("\nContainer test complete")

            except Exception as e:
                print(f"❌ Error processing {image}:{tag}: {str(e)}")
                success = False
                break

            finally:
                self.cleanup_container(container_name)

        if success:
            self.run_command("sudo ctr-starlight optimizer off", "Stopping optimizer")
            self.run_command("sudo ctr-starlight report --profile myproxy", 
                           "Reporting optimization results")

        return success

    def convert_and_optimize(self, image: str, tags: List[str]) -> bool:
        print(f"\n=== Processing {image.upper()} ===")
        success = True

        # 変換が必要なタグを特定
        tags_to_convert = []
        for tag in tags:
            starlight_tag = f"{tag}-starlight"
            if self.check_registry_tag(image, starlight_tag):
                print(f"ℹ️ Image {image}:{starlight_tag} already exists in registry, skipping conversion")
            else:
                tags_to_convert.append(tag)

        # 必要なタグの変換を実行
        for tag in tags_to_convert:
            if not self.convert_image(image, tag):
                print(f"❌ Failed to convert {image}:{tag}")
                return False

        # 最適化とレポート生成を実行（全タグに対して）
        if not self.optimize_and_report(image, tags):
            print(f"❌ Failed to optimize and report for {image}")
            return False

        return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <cloud_host>")
        print("Example: python script.py cloud.cluster.local:5000")
        sys.exit(1)

    cloud_host = sys.argv[1]
    optimizer = DockerTagOptimizer(cloud_host)

    images = ['nginx', 'ubuntu', 'mysql', 'postgres', 'node', 
              'redis', 'alpine', 'mongo', 'python', 'centos']
    
    print("=== Docker Image Optimization Tool ===")
    print(f"Cloud Host: {cloud_host}\n")
    
    image_tags = {}
    total_valid_images = 0
    
    print("Analyzing available tags for all images...")
    for image in images:
        print(f"\n{image.upper()}:")
        tags = optimizer.fetch_docker_tags(image)
        
        if tags:
            filtered_tags = optimizer.filter_unsuitable_tags(tags)
            recommended_tags = optimizer.find_consistent_tags(filtered_tags)
            
            if recommended_tags:
                image_tags[image] = recommended_tags
                total_valid_images += 1
                
                registry_tags = optimizer.fetch_registry_tags(image)
                
                for i, tag in enumerate(recommended_tags, 1):
                    version_nums, prefix, suffix = optimizer.parse_version(tag)
                    suffix_info = f" [{suffix}]" if suffix else ""
                    starlight_tag = f"{tag}-starlight"
                    cache_status = "cached" if starlight_tag in registry_tags else "not cached"
                    print(f"  {i}. {tag}{suffix_info} ({cache_status})")
                
                version_depth = len(optimizer.parse_version(recommended_tags[0])[0])
                pattern = "x" + ".x" * (version_depth - 1)
                print(f"  Version pattern: {pattern}")
            else:
                print("  No suitable consistent tag set found")
        else:
            print("  Unable to fetch tags")
    
    if not image_tags:
        print("\nNo valid images found to process.")
        sys.exit(1)
    
    print(f"\nFound {total_valid_images} images to process.")
    print("\nThe following operations will be performed for each image:")
    print("1. Convert images to Starlight format (if not cached)")
    print("2. Pull and test converted images")
    print("3. Collect and report optimization data")
    print("\nDo you want to continue? [y/N] ", end='')
    
    response = input().lower()
    if response != 'y':
        print("Operation cancelled by user")
        return
    
    print("\nStarting conversion and optimization process...")
    
    for image, tags in image_tags.items():
        if optimizer.convert_and_optimize(image, tags):
            print(f"✅ Successfully processed {image}")
        else:
            print(f"❌ Failed to process {image}")
            print("Continuing with next image...")

    print("\nOptimization process completed.")

if __name__ == "__main__":
    main()