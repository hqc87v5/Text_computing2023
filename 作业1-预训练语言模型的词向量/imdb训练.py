import os
import shutil
import requests
import tempfile
from tqdm import tqdm
from typing import IO
from pathlib import Path

import re
import six
import string
import tarfile
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
            #tempfile.NamedTemporaryFile() 创建有名字的、在文件系统中可见的临时文件
            http_get(url, temp_file)
            temp_file.flush() #关闭文件输出流
            temp_file.seek(0)#seek(0)让指针定位到开头，就可以看到打印输出的内容了。
            with open(cache_path, 'wb') as cache_file:
                shutil.copyfileobj(temp_file, cache_file)
    return cache_path

imdb_path = download('aclImdb_v1.tar.gz', 'https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/aclImdb_v1.tar.gz')

class IMDBData():
    #IMDB数据集加载器，加载IMDB数据集并处理为一个python迭代对象
    label_map={
        "pos":1,
        "neg":0
    }
    def __init__(self,path,model="train"):
        self.mode = model
        self.path = path
        self.docs,self.labels = [],[]

        self._load("pos")
        self._load("neg")
    def _load(self, label):
        pattern = re.compile(r"aclImdb/{}/{}/.*\.txt$".format(self.mode,label))
        #加载数据到内存
        # tarfile解决zip压缩包的创建、读取、写入
        with tarfile.open(self.path) as tarf:
            tf = tarf.next()
            while tf is not None:
                if bool(pattern.match(tf.name)):
                    self.docs.append(str(tarf.extractfile(tf).read().rstrip(six.b("\n\r")).
                                         translate(None,six.b(string.punctuation)).lower()).split())
                    self.labels.append([self.label_map[label]])
                    tf = tarf.next()

    def __getitem__(self, idx):
        return self.docs[idx],self.labels[idx]
    def __len__(self):
        return len(self.docs)

imdb_train = IMDBData(imdb_path,'train')
print(len(imdb_train))
