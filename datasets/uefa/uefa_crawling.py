from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

url = "./datasets/uefa/uefa.html"
page = open(url, 'rt', encoding = 'utf-8').read()
soup = BeautifulSoup(page, 'html.parser') 
t = soup.find_all(attrs= {"class" : "sc-pk-table-row-h sc-pk-table-row-s pk-table--row hydrated"})

store_tmp = []
team_store = []

for idx, t in enumerate(t): 
    cnt = 0
    sum = 0
    for iter in range(2, 13):
        if (iter == 2):
            team =  t.contents[iter].contents[1].contents[0].contents[1].contents[0]
            team_store.append(team)
        
        if(3 <= iter <= 12):
            value = t.contents[iter].contents[1]
            if(value != "-"):
                cnt += 1
                sum += float(value)
            store_tmp.append(value)
        
    if(cnt != 0):
        mean = round(sum / cnt, 3)

    for i in range(idx * 10, idx * 10 + 10):
        if(store_tmp[i] == "-"):
            store_tmp[i] = mean

numpy_tmp = np.array(store_tmp).reshape(-1, 10)

result = pd.DataFrame(numpy_tmp, columns = ["2012-2013", "2013-2014", "2014-2015", "2015-2016", "2016-2017", "2017-2018", "2018-2019", "2019-2020", "2020-2021", "2021-2022"], index = team_store)

result.to_csv('./datasets/uefa/uefa_result.csv', sep = ',')