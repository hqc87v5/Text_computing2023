import os
import shutil
import requests
import tempfile
from tqdm import tqdm
from typing import IO
from pathlib import Path

# 指定保存路径为 `home_path/.mindspore_examples`
cache_dir = Path.home() / '.mindspore_examples'

def http_get(url: str, temp_file: IO):
    """使用requests库下载数据，并使用tqdm库进行流程可视化"""
    req = requests.get(url, stream=True)
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit='B', total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()

def download(file_name: str, url: str):
    """下载数据并存为指定名称"""
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_path = os.path.join(cache_dir, file_name)
    cache_exist = os.path.exists(cache_path)
    if not cache_exist:
        with tempfile.NamedTemporaryFile() as temp_file:
            http_get(url, temp_file)
            temp_file.flush()
            temp_file.seek(0)
            with open(cache_path, 'wb') as cache_file:
                shutil.copyfileobj(temp_file, cache_file)
    return cache_path

imdb_path = download('aclImdb_v1.tar.gz', 'https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/aclImdb_v1.tar.gz')
