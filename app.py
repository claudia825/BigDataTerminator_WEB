import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from konlpy.tag import Okt
import pandas as pd
import numpy as np
from kiwipiepy import Kiwi, Option
from gensim.models import FastText
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fasttext
import re
import io
import json
import csv
import openpyxl
import pandas as pd
import numpy as np
import ast
from konlpy.tag import Komoran
from IPython.core.display import display, HTML
import streamlit as st


def fileopen(data):
    
    with open(data, 'r', encoding='utf-8') as file:
        
        text = file.read()
        
        splitdata = text.split()
        
        splitdata = list(dict.fromkeys(splitdata))
 
    return splitdata
 
 
def crawl():
    st.write("데이터 크롤링을 시작할 날짜를 입력해주세요.")
    st.write("15일치 데이터를 크롤링합니다. (test)")
    date = st.text_input("시작 날짜 입력. (ex. 20210920) : ")

    for i in range(0,16):
    
        for j in range(1,3):
            browser.get("https://pann.nate.com/talk/ranking/d?stdt=" + str(date) + "&page=" + str(j))
            soup = BeautifulSoup(browser.page_source, 'html.parser')

            first_list = soup.find('div', {'class': 'cntList'}).findAll('li')

        for li in first_list: 
            f_link = li.findAll('a')
            for a in f_link:
                real_link = 'https://pann.nate.com' + a.get('href') 
            links.append(real_link)
        
        for li in first_list:
            f_title = li.findAll('dl')
            for dl in f_title:
                t = dl.find('a')
                real_title = t.get('title')
                real_title = spacing(real_title)
                real_title = re.sub('[ㄱ-ㅎㅏ-ㅣa-zA-Z0-9]', '', real_title)
                real_title = re.sub('[\xa0|0xed]', '', real_title)
                real_title = re.sub('[-=+_★♥♡,#/\?:╋^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》—;]','',real_title)
            titles.append(real_title)
            
        date = int(date) + 1

        txt = []
    
    for i in links:
        try: 
            res = requests.get(i)
            res.raise_for_status()
            res.encoding = None            
            html2 = res.text
            
            soup = BeautifulSoup(html2, 'html.parser')
            contentArea = soup.find("div", {"class" : "viewarea"})            
            parags = contentArea.findAll("div", {"id" : "contentArea"})

            content = ""

            for parag in parags:
                content += parag.text
            content = re.sub('[ㄱ-ㅎㅏ-ㅣa-zA-Z0-9]', '', content)
            content = re.sub('[&nbsp;|\n|\t|\r]', '', content)
            content = re.sub('[\xa0]', '', content)
            content = re.sub('[-=+_★♥♡,#/\?:╋^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》—;]', '', content)
            content = re.sub('[^ ㄱ-ㅣ가-힣]','',content)
            #content = spacing(content)
            
            txt.append(content)
                
        except HTTPError as e:
            txt.append('')
        except URLError as e:
            txt.append('')
        except AttributeError as e:
            txt.append('')


    okt = Okt() 

    title_morphs = []
    txt_morphs = []
        
    for i in titles:
        title_morphs.append(okt.morphs(i))
        
    for i in txt:
        txt_morphs.append(okt.morphs(i))
        

    nate_dict = {
        '제목' : titles,
        '본문' : txt
    }

    df = pd.DataFrame(nate_dict) 
    df.to_csv('natepann.csv', index=False, encoding="utf-8-sig")

    
    morphs_dict = {
        '제목 형태소' : title_morphs,
        '본문 형태소' : txt_morphs

    }


    df2 = pd.DataFrame(morphs_dict)
    df2.to_csv('natepann_Morphs.csv', index=False, encoding="utf-8-sig")

    browser.close()

    df.to_csv('natepann.txt', sep = '\t', index = False)
    df = pd.read_csv('natepann_Morphs.csv')
    df.to_csv('natepann_Morphs.txt', sep = '\t', index = False)

def newwords():
    # 새로운 용어 추출 - 단순 비교
    NewList1 = fileopen('natepann.txt')
    
    NewList2 = fileopen('SentiWord_Dict.txt')
    
    NewList3 = fileopen('sejong.txt')

    wordlist = list((set(NewList1)-set(NewList2))-set(NewList3))

    #키위
    kiwi = Kiwi()

    Kiwi(
        num_workers=0, 
        model_path=None, 
        load_default_dict=True, 
        integrate_allomorph=True
    )

    inputs = wordlist

    
    st.write(kiwi.extract_words(inputs, min_cnt=1, max_word_len=10, min_score=0.25, pos_score=-3.0, lm_filter=True))


if __name__ == '__main__':

    st.title("온라인 커뮤니티 특화 감성 사전 구축을 위한 새로운 용어 극성값 분석 시스템")
    st.header("BigDataTerminator")

    crawl()
    newwords()