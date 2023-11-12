#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

def fetchUrl(pid, uid, max_id):
    
    url = "https://weibo.com/ajax/statuses/buildComments"
    
    headers = {
 
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
    }
    
    params = {
 
        "flow" : 0,
        "is_reload" : 1,
        "id" : pid,
        "is_show_bulletin" : 2,
        "is_mix" : 0,
        "max_id" : max_id,
        "count" : 20,
        "uid" : uid,
    }

    r = requests.get(url, headers = headers, params = params)
    return r.json()

def parseJson(jsonObj):

    data = jsonObj["data"]
    max_id = jsonObj["max_id"]

    commentData = []
    for item in data:
        # 评论id
        comment_Id = item["id"]
        # 评论内容
        content = BeautifulSoup(item["text"], "html.parser").text
        # 评论时间
        created_at = item["created_at"]
        # 点赞数
        like_counts = item["like_counts"]
        # 评论数
        total_number = item["total_number"]
        
        # 评论者 id，name，city
        user = item["user"]
        userID = user["id"]
        userName = user["name"]
        userCity = user["location"]
        
        dataItem = [comment_Id, created_at, userID, userName, userCity, like_counts, total_number, content]
        print(dataItem)
        commentData.append(dataItem)
        
    return commentData, max_id

def save_data(data, path, filename):
    
    if not os.path.exists(path):
        os.makedirs(path)

    dataframe = pd.DataFrame(data)
    dataframe.to_csv(path + filename, encoding='utf_8_sig', mode='a', index=False, sep=',', header=False )

if __name__ == "__main__":
    
    pid = 4759011825814260      # 微博id，固定
    uid = 2286908003            # 用户id，固定
    max_id = 0
    path = "D:/"           # 保存的路径
    filename = "comments.csv"   # 保存的文件名
    
    csvHeader = [["评论id", "发布时间", "用户id", "用户昵称", "用户城市", "点赞数", "回复数", "评论内容"]]
    save_data(csvHeader, path, filename)

    while(True):
        html = fetchUrl(pid, uid, max_id)
        comments, max_id = parseJson(html)
        save_data(comments, path, filename)
        # max_id 为 0 时，表示爬取结束
        if max_id == 0:
            break;

