from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
from datetime import date
import time
import shutil

original = r'C:\Users\chuac\Documents\UM\WQD7005\Project\fullData\covid.csv'

url = 'https://google.com/covid19-map/'
driver = webdriver.Chrome(executable_path=r'C:\Users\chuac\Documents\UM\WQD7005\Project\chromedriver.exe')
driver.get(url)
driver.execute_script("window.scrollTo(0, 400);")
div = driver.find_element_by_xpath('//*[@id="yDmH0d"]/c-wiz/div/div[2]/div[2]/div[4]/div/div/div[2]/div/div[1]/table')
driver.execute_script('arguments[0].scrollTop = arguments[0].scrollHeight', div)
time.sleep(10)

#Parse html content
soup = BeautifulSoup(driver.page_source, 'html.parser')
table = soup.find('table')
df = pd.read_html(str(table))

driver.close()

#Adding Date column 
df[0]['Date'] =  date.today().strftime("%Y-%m-%d")
cols = ['Date','Location','Confirmed','Deaths','Recovered']
df2 = df[0][cols]

#Saving dataframe to csv file
df2.to_csv(original,mode='a', index=False, header=False)
print("Done scraping data for COVID-19!")

## Web crawling for twitter post on COVID-19 issue
import twitter
import numpy as np 
import json 
import pandas as pd

original2 = r'C:\Users\chuac\Documents\UM\WQD7005\Project\fullData\twitter.txt'
target2 = r'C:\Users\chuac\Documents\UM\WQD7005\Project\fullData\twitter_bk_' + date.today().strftime("%Y%m%d")+ '.txt'

# initialize api instance
twitter_api = twitter.Api(consumer_key='XXXXX',
                        consumer_secret='XXXXX',
                        access_token_key='XXXXX',
                        access_token_secret='XXXXX')

search_keyword = 'covid-19 OR wuhanvirus OR coronovirus -filter:retweets'

try:
    tweets = twitter_api.GetSearch(term = search_keyword,lang='en',result_type='recent',count = 100)
except:
    print("Unfortunately, something went wrong..")
    
tw_data = pd.DataFrame(tweets)

tw_data = pd.DataFrame(columns = ['Date','Line'])
tw_data['Line'] = tweets
tw_data['Date'] = date.today().strftime("%Y-%m-%d") + "\t"
f=open(target2,'ab')
np.savetxt(target2, tw_data.values, fmt = "%s")
f.close()
print("Done scraping twitter data for COVID-19!")
