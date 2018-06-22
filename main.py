# -*- coding: utf-8 -*-

"""
    项目名称：电影海报主色调聚类分析
    项目参考：http://blog.nycdatascience.com/student-works/using-python-and-k-means-to-find-the-colors-in-movie-posters/
"""
from bs4 import BeautifulSoup
import requests
import re
import os
import skimage.io
import skimage.transform
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

# 路径申明
save_path = './images/'
kmeans_results_path = './kmeans_results'
# 主题色个数
n_main_color = 10
# 是否下载图片
is_download = True


def download_posters(image_type):
    """
        下载电影海报并保存到本地
    """
    print('正在下载{}类型的电影海报...'.format(image_type))
    query = 'movie 2016 ' + image_type + ' poster'
    url = 'http://global.bing.com/images/search?q=' + query + '&FORM=HDRSC2'
    soup = BeautifulSoup(requests.get(url).text, 'lxml')
    img_src_list = [a['src'] for a in soup.find_all('img', {'src': re.compile('mm.bing.net')})]
    for i, img_src in enumerate(img_src_list):
        img_data = skimage.io.imread(img_src)
        if img_data is not None:
            save_img_name = image_type + '_' + str(i + 1) + '.jpg'
            skimage.io.imsave(os.path.join(save_path, save_img_name), img_data)
            print('已下载{}张'.format(i + 1))
        else:
            print('该图像无效', img_src)
    print()


def proc_img(img_filename):
    """
        读取海报，并运行K-Means找出10个主要颜色
    """
    img = skimage.io.imread(os.path.join(save_path, img_filename))
    # 调整图片大小至 200 x 200
    resized_img = skimage.transform.resize(img, (200, 200))
    img_data = resized_img.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_main_color)
    kmeans.fit(img_data)
    centers = kmeans.cluster_centers_

    # 将每个像素值扩展到20x20x3的矩形框中，用于保存查看
    color_block_size = 20
    main_color_img = np.zeros((color_block_size * n_main_color, color_block_size, 3))
    for i, center in enumerate(centers):
        main_color_img[i * color_block_size: (i + 1) * color_block_size, :, :] = center
    skimage.io.imsave(os.path.join(kmeans_results_path, img_filename), main_color_img)

    # 保存数据到一行dataframe中
    kmeans_result_df = pd.DataFrame()
    kmeans_result_df['image name'] = [img_filename]
    kmeans_result_df['movie type'] = [img_filename.split('_')[0]]
    for i, center in enumerate(centers):
        rgb_val = skimage.img_as_ubyte(center)
        kmeans_result_df['color{}_R'.format(i + 1)] = [rgb_val[0]]
        kmeans_result_df['color{}_G'.format(i + 1)] = [rgb_val[1]]
        kmeans_result_df['color{}_B'.format(i + 1)] = [rgb_val[2]]
    return kmeans_result_df


def run_main():
    """
        主程序
    """
    # 爬取电影海报
    movie_types = ['horror', 'comedy', 'animation', 'action']

    if is_download:
        for movie_type in movie_types:
            download_posters(movie_type)

    # 读取每张海报，并运行K-Means找出每张海报的10个主要颜色，并构建数据集
    img_filename_list = os.listdir(save_path)
    result_df = pd.DataFrame()
    for img_filename in img_filename_list:
        print('正在处理', img_filename)
        kmeans_result_df = proc_img(img_filename)
        result_df = result_df.append(kmeans_result_df, ignore_index=True)

    result_df.to_csv('./kmeans_results.csv', index=False)

if __name__ == '__main__':
    run_main()
