from pandas.core.strings import copy
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from konlpy.tag import Okt
import pandas as pd
import numpy as np
from kiwipiepy import Kiwi
from gensim.models import FastText
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np
import ast
from konlpy.tag import Komoran
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
from bs4 import BeautifulSoup
from selenium.webdriver import Chrome
from selenium.webdriver.common.keys import Keys
import streamlit as st
from matplotlib import font_manager, rc
import platform


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
    kiwi.load_user_dictionary('new.txt')
    
    wlist = []
    wlists =list(kiwi.extract_words(inputs, min_cnt=1, max_word_len=10, min_score=0.25, pos_score=-3.0, lm_filter=True))
    for i in range(len(wlists)):
        temp_str = wlists[i][0].__str__()
        wlist.append(temp_str)
    
    st.write(wlist)


def an(new):
    loaded_model = FastText.load("fasttext")
    global Similar
    Similar = []
    similar_word_list = list(loaded_model.wv.most_similar(str(new), topn=10)) 
    for i in range(len(similar_word_list)):
        temp_str = similar_word_list[i][0].__str__()
        Similar.append(temp_str)
    
    st.write(Similar)
    return Similar

def extract_word(fkey,key):
    
    final = pd.DataFrame(columns = ['contents'])

    f = pd.read_csv('natepann_data.csv')

    is_key = (f['본문'].str.contains(fkey, na=False) | f['제목'].str.contains(fkey, na=False)) & (f['본문'].str.contains(key, na=False) | f['제목'].str.contains(key, na=False))

    f_key = f[is_key]
    
    if f_key.empty:
        pass
    else: 
        f_key = f_key.reset_index()

        final_key = pd.DataFrame(columns = ['tokenized'])

        komoran = Komoran()

        for i in range(len(f_key)):
            try:
                st = f_key.loc[i, '제목']
                final_key.loc[i, 'tokenized'] = str(st)
                ct = f_key.loc[i, '본문']
                final_key.loc[i, 'tokenized'] += str(ct)
                
            except KeyError:
                print("error")

        keyword_content = pd.DataFrame(columns = ['content','positive', 'negative', 'neutral'])

        for i in range(len(final_key)):
            try:
                fc = final_key.loc[i, 'tokenized']
                
                if i == 0:
                    keyword_content.loc[0,'content'] = fc
                else:
                    keyword_content.loc[0,'content'] += fc
                
            except KeyError:
                print("error")

        #st.dataframe(keyword_content, columns = ['content'])

        # 감성사전에서 감성분석
        f = open('SentiWord_Dict.txt', 'r', -1, 'utf-8')
        lines = f.readlines()

        score_dict = []

        for line in lines:
            line_splited = line.split()
            score = int(line_splited[-1])
            word = ''
            for frac in line_splited[:-1]:
                word = word + ' ' + frac
            
            word = word[1:]
            score_dict.append([word, score])
            
        keyword_content['positive'] = 0
        keyword_content['negative'] = 0
        keyword_content['neutral'] = 0

        if keyword_content.empty:
            #print("사전과 비교할 조건에 맞는 내용 없음")
            pass
        else:
            keyword_content.loc[0, 'content1'] = str(komoran.nouns(keyword_content.loc[0, 'content']))


            stopwords = ['하다', '없다', '있다', '되다', '아니다', '같다', '이다', '않다', '그렇다', 
                        '이렇다', '싶다', '다', '것', '그', '이', '거', '니다']


            pos_list = ast.literal_eval(keyword_content.loc[0, 'content1'])

            final = []

            for j in range(len(pos_list)):
                if pos_list[j] not in stopwords:
                    final.append(pos_list[j])

            keyword_content.loc[0, 'content1'] = str(final)

            # 감성사전에서 감성분석
            f = open('SentiWord_Dict.txt', 'r', -1, 'utf-8')
            lines = f.readlines()

            score_dict = []

            for line in lines:
                line_splited = line.split()
                score = int(line_splited[-1])
                word = ''
                for frac in line_splited[:-1]:
                    word = word + ' ' + frac

                    word = word[1:]
                    score_dict.append([word, score])

            keyword_content.insert(0, "Keyword", key)

            tokens = ast.literal_eval(keyword_content.loc[0, 'content1'])

            try:
                for token in tokens:
                    for dict_word in score_dict:
                        if dict_word[0] == token:
                            if dict_word[1] > 0:
                                keyword_content.loc[0, 'positive'] += dict_word[1]
                            elif dict_word[1] < 0:
                                keyword_content.loc[0, 'negative'] += dict_word[1]
                            else:
                                keyword_content.loc[0, 'neutral'] += 1 #중립어는 개수 세기    
            except KeyError:
                print("error")
        
        #keyword_content[keyword_content]
        df_reset=keyword_content.set_index('Keyword')
        df_reset[df_reset.columns.difference(['content1'])]

        global pos, neg, neu
        pos.append(int(keyword_content.loc[0,'positive']))
        neg.append(int(keyword_content.loc[0,'negative']))
        neu.append(int(keyword_content.loc[0,'neutral']))


def NLP(a,b,c) :
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)

    pos = a
    neg = b
    nneg = str(neg).replace("-", "")
    neu = c 
    
    p = pos / (pos + int(nneg))*100
    n = int(nneg) / (pos + int(nneg))*100
    total = p - n

    if total <= 10 and total >= -10:
        Senti = 0
    elif total < -10 and total >= -50:
        Senti = -1
    elif total < -50:
        Senti = -2
    elif total > 10 and total<= 50:
        Senti = 1
    elif total > 50:
        Senti = 2

    size = [pos, nneg, neu]

    label = ['Positive','Negative', 'Neutral']
    colors = ['#FBF9A9', '#FFD9D9', '#8fd9b6']

    plt.subplots()
    plt.axis('equal')
    plt.rc('font', size=10)
    plt.pie(x=size, startangle=200, colors = colors, shadow = True, labels=label, autopct='%.2f%%')
    plt.rc('legend', fontsize=8)
    plt.legend(loc="upper right")

    plt.title(str(fkey) + "의 극성값 = " + str(Senti))
    #plt.show()
    st.pyplot(plt)

    st.subheader("Copy and Paste for Senti_Dict :")
    st.info(str(fkey) + "       " + str(Senti))

if __name__ == '__main__':

    #footer 수정
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden; }
                footer {visibility: hidden;}
                footer:after {visibility: visible; content:"BigDataTerminator";}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

   
    st.title("신조어 극성값 분석 시스템")

    st.subheader("BigDataTerminator")
    #st.subheader("온라인 커뮤니티 특화 감성 사전 구축을 위한 신조어의 극성값 분석 시스템")

    st.write("데이터 크롤링을 시작할 날짜를 입력해주세요.")
    st.write("6개월 데이터를 크롤링합니다.")
    date = st.text_input("시작 날짜 입력. (ex. 20210920) : ", help=None)

    # fin = 0

    #웹에서 바로 크롤링하면 2일치만 해도 너무 느림... 
    #crawl(date, fin)

    # if fin == 1:
    #     st.write("크롤링 완료")
       
    #     newwords()

    #     st.subheader("연관 용어 추출")

    pos = []
    neg = []
    neu = []
    global key, fkey

    
    if len(date) >= 8:
        st.write('크롤링 완료!')
        st.subheader("신조어 추출")
        newwords()

        new = st.text_input("""분석할 "신조어"를 입력하세요. : """)

        if st.button('신조어 분석'):
            fkey = new
            st.subheader("'" + new + "' 의 연관 용어 추출")
            an(new)
            k = 0
            k += 1
            
            if k > 0:
                st.subheader("신조어 : '" + new +  "' 의 감성 분석")
                with st.spinner("연관용어 분석 중..."):
                    for i in range(len(Similar)): 
                        key = Similar[i]
                        extract_word(fkey,key)
                st.success("분석 완료")

                st.write(" ")
                st.subheader("분석 결과 :")
                NLP(sum(pos),sum(neg),sum(neu))


            