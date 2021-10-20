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
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
from bs4 import BeautifulSoup
from selenium.webdriver import Chrome
from selenium.webdriver.common.keys import Keys
from IPython.core.display import display, HTML
from pykospacing import spacing
import streamlit as st


def crawl(date, fin):
    if len(date) == 8:
    
        browser = Chrome()

        titles = []
        links = []

        for month in range(3,10):
            for date in range(1,32):

                for j in range(1,3):
                    browser.get("https://pann.nate.com/talk/ranking/d?stdt=2021" + str(month).zfill(2) + str(date).zfill(2)+"&page=" + str(j))
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
                        real_title = re.sub('[^ ㄱ-ㅣ가-힣]','',real_title)
                    titles.append(real_title)

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
                    content = re.sub('[^ ㄱ-ㅣ가-힣]','',content)

                    txt.append(content)
                        
                except HTTPError as e:
                    txt.append('')
                except URLError as e:
                    txt.append('')
                except AttributeError as e:
                    txt.append('')

            browser.close()
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

            df.to_csv('natepann_data.txt', sep = '\t', index = False)
            
        
    fin = 1
    return fin


def fileopen(data):
    
    #파일 불러오기 
    with open(data, 'r', encoding='utf-8') as file:
        
        text = file.read()
        
        splitdata = text.split()
        
        #리스트의 중첩된 부분 삭제하기 
        splitdata = list(dict.fromkeys(splitdata))
 
    return splitdata

def newwords():
    # 새로운 용어 추출 - 단순 비교
    NewList1 = fileopen('natepann_data.txt')
        
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


def an():
    loaded_model = FastText.load("fasttext")
    Similar = []
    similar_word_list = list(loaded_model.wv.most_similar("스우파", topn=10)) 
    for i in range(len(similar_word_list)):
        temp_str = similar_word_list[i][0].__str__()
        Similar.append(temp_str)
    
    st.write(Similar)



if __name__ == '__main__':

    st.header("온라인 커뮤니티 특화 감성 사전 구축을 위한 새로운 용어 극성값 분석 시스템")

    st.subheader("BigDataTerminator")

    st.write("데이터 크롤링을 시작할 날짜를 입력해주세요.")
    st.write("TEST) 15일치 데이터를 크롤링합니다.")
    date = st.text_input("시작 날짜 입력. (ex. 20210920) : ")
    # fin = 0

    #웹에서 바로 크롤링하면 2일치만 해도 너무 느림... 
    #crawl(date, fin)

    # if fin == 1:
    #     st.write("크롤링 완료")
       
    #     newwords()

    #     st.subheader("연관 용어 추출")


    if len(date) >= 8:
        st.write("크롤링 완료")

        newwords()

        st.subheader("연관 용어 추출")

        an()
