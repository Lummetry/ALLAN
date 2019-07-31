import nltk
import requests
import pandas as pd

from bs4 import BeautifulSoup
from libraries.logger import Logger
from nltk.tokenize import RegexpTokenizer


#Expand dataset by: using website archive: hotnews.ro/arhiva/2018-01-01 to get links     
      
class Spider(object):
  def __init__(self, logger, domain, url_source, urls_tag, data_tag):
    self.logger = logger
    self.domain_name = domain
    self.domain_url = 'https://' + domain
    self.urls_tag = urls_tag
    self.data_tag = data_tag
    
    self.site_request = requests.get(self.domain_url)
    self.page = self.site_request.content
    
    self.article_sources = self.get_news_urls()
    self.logger.P('Found {} articles on {}'.format(len(self.article_sources), self.domain_url + url_source))
    
    self.documents, self.labels = self.get_labeled_documents()
    self.process_labels()
    
  def validate_url(self, url):
    request = requests.get(url)
    
    if request.status_code == 200:
        self.logger.P('Website exists')
        return True
    else:
        self.logger.P('URL {} does not exist'.format(url))
        return False
        
  
  def get_news_urls(self):
    url_soup = BeautifulSoup(self.page, 'html.parser')
    
    list_of_articles = url_soup.find_all(self.urls_tag[0] , class_=self.urls_tag[1])
    
    if not list_of_articles:
      self.logger.P("urls_tag [{}] is not yielding links".format(self.urls_tag))
      
    list_of_links = []
    for i in list_of_articles:
      link = i.find('a')['href']
      
      if(link[:4] == 'http'):
        list_of_links.append(link)
      
      else:
#        if(self.validate_url(self.domain_url + link) == True):
        list_of_links.append(self.domain_url + link)
    
    return list_of_links
      
  def get_text_and_label(self, url):
    news_request = requests.get(url)
    self.logger.P('URL {} returned status code {}'.format(url, news_request.status_code))
    
    news_source = BeautifulSoup(news_request.content, 'html.parser')
    
    if self.data_tag[1] == '':
      self.data_tag[1] = None
    
    if self.data_tag[3] == '':
      self.data_tag[3] = None
    
    news_article = news_source.find(self.data_tag[0], 
                                    class_=self.data_tag[1])
    
    if news_article is None:
      return [], []
    
    news_text = news_article.find_all(self.data_tag[2], 
                                      class_=self.data_tag[3]) 
    
    text = ''
    for i in news_text:
      text = text + i.get_text()
      

    self.logger.P('Found <{} class={}> tag in the webpage, found {} number of <{} class={}> tag(s) containing {} words'.format(self.data_tag[0], self.data_tag[1], len(news_text) ,self.data_tag[2], self.data_tag[3], len(text.split())))
  
    metatags = news_source.find_all('meta',attrs={'name':'keywords'})
    
    labels = []
    for tag in metatags:
      s = tag.get('content')
      s = self.tokenizer.tokenize(s.lower())
      labels.append(s)      
    
    #flatten list
    labels = [item for sublist in labels for item in sublist]
    #remove duplicates
    labels = list(set(labels))
    
    return text, labels
  
  def get_labeled_documents(self):
    documents = []
    labels = []
    
    self.tokenizer = RegexpTokenizer(r'\w+')

    for i in self.article_sources[:2]:
      doc, lbl = self.get_text_and_label(i)
      documents.append(doc)
      labels.append(lbl)
      
    return documents, labels
  
  def process_labels(self):
    self.all_labels = [item for sublist in self.labels for item in sublist]
    
    
    print(self.all_labels)
    df = pd.DataFrame(columns=['labels'])
    df.labels = self.all_labels
    df_distrib = df.describe()
    
    self.logger.P("Distribution of lengths between start and end:\n{}".format(df_distrib.to_string()))
    return 
    
    
  
  
if __name__ == '__main__':
  logger = Logger(lib_name='DOC-COLLECTOR', 
                config_file='./tagger/crawler/config_crawler.txt')

  Digi24 = Spider(logger, 'digi24.ro','stiri/actualitate', ('h4','article-title'), ['article', 'article-story', 'p',''])
  Digi24.get_labeled_documents()
  
  Hotnews = Spider(logger, 'hotnews.ro','', ('h2','article_title'), ['div','articol_render','div',''])
  Hotnews.get_labeled_documents()
  