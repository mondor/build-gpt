import os
import time
import requests
from multiprocessing import Pool
from tqdm import tqdm

BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542
DATA_DIR = '../../climbmix'


def download_single_file(index):
    filename = f'shard_{index:05d}.parquet'
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f'Skipping {filepath} already exists')
        return True
    url = f'{BASE_URL}/{filename}'
    print(f'Downloading {url}...')
    max_attempts = 6
    for attempt in range(max_attempts):
        try:
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()
            temp_path = filepath + '.tmp'
            total = int(resp.headers.get("content-length", 0))
            with open(temp_path, 'wb') as f, tqdm(desc=temp_path, total=total, unit='iB', unit_scale=True,
                                                  unit_divisor=1024, ) as bar:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        size = f.write(chunk)
                        bar.update(size)
            os.rename(temp_path, filepath)
            print(f'Successfully downloaded {filepath}')
            return True
        except Exception as e:
            print(f'Attempt {attempt}/{max_attempts} failed: {e}')
            for path in [filepath + '.tmp', filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    return False


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-files', type=int, default=170)
    args = parser.parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)
    num_train = min(args.num_files, MAX_SHARD)
    ids = list(range(num_train))
    print(f'Downloading {len(ids)} shards to {DATA_DIR}')
    nprocs = max(1, os.cpu_count() // 2)
    with Pool(processes=nprocs) as pool:
        results = pool.map(download_single_file, ids)
    print(f'Done! {sum(results)}/{len(ids)} shards downloaded')
