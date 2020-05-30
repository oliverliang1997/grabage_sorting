# -*- coding: utf-8 -*-
#writeTxt(dirpath,txtName,img_name)


import re
import sys
import urllib
import os
import requests

dirpath ='../images/27/'

def get_onepage_urls(onepageurl):
    if not onepageurl:
        print('end')
        return [], ''
    try:
        html = requests.get(onepageurl)
        html.encoding = 'utf-8'
        html = html.text
    except Exception as e:
        print(e)
        pic_urls = []
        fanye_url = ''
        return pic_urls, fanye_url
    pic_urls = re.findall('"objURL":"(.*?)",', html, re.S)
    fanye_urls = re.findall(re.compile(r'<a href="(.*)" class="n">下一页</a>'), html, flags=0)
    fanye_url = 'http://image.baidu.com' + fanye_urls[0] if fanye_urls else ''
    return pic_urls, fanye_url

m = 0
word_id = 1

def down_pic(pic_urls):
    for i, pic_url in enumerate(pic_urls):
        try:
            pic = requests.get(pic_url, timeout=15)
            img_name ='fimg_' + str(i + m) + '.jpg'
            filename = os.path.join(dirpath, img_name)
            with open(filename, 'wb') as f:
                f.write(pic.content)
                print('success%s: %s' % (str(i + 1), str(pic_url)))
            txtName = 'fimg_' + str(i + m) + '.txt'
        except Exception as e:
            print('failed%s: %s' % (str(i + 1), str(pic_url)))
            print(e)
            continue
            
def writeTxt(dirpath, txtName, img_name):
    filename = os.path.join(dirpath, txtName)
    file = open(filename, 'w')
    line = img_name + ', ' + str(word_id)
    print(line)
    file.write(line)
    file.close()
    return True

if __name__ == '__main__':
    keyword = '牙签'
    url_init_first = r'http://image.baidu.com/search/flip?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1497491098685_R&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&ctd=1497491098685%5E00_1519X735&word='
    url_init = url_init_first + urllib.parse.quote(keyword, safe='/')
    all_pic_urls = []
    onepage_urls, fanye_url = get_onepage_urls(url_init)
    all_pic_urls.extend(onepage_urls)

    fanye_count = 0
    while 1:
        onepage_urls, fanye_url = get_onepage_urls(fanye_url)
        fanye_count += 1
        if fanye_url == '' and onepage_urls == []:
            break
        all_pic_urls.extend(onepage_urls)

    down_pic(list(set(all_pic_urls)))


