import re
import nltk
import requests
import pandas as pd

from tqdm import tqdm
from bs4 import BeautifulSoup
from collections import Counter
from libraries.logger import Logger
from nltk.tokenize import RegexpTokenizer

def flatten_list(a):
  return [item for sublist in a for item in sublist]

def is_number_larger_than_x(number, x):
  try:
    if int(number) > x:
      return True
    else:
      return False
  except ValueError:
    return False
    

#Expand dataset by: using website archive: hotnews.ro/arhiva/2018-01-01 to get links     
      
#Spider takes main domain, 
#             a page from where it pulls articles,
#             the html tag where links to articles are on the page where it pulls articles from,
#             the html tags where the text data is found: list[article tag, article class, text data tag, text data class]

class Spider(object):
  def __init__(self, logger, DEBUG,
               domain,
               archive,
               url_cycle_source,
               cycle_range,
               urls_tag,
               data_tag,
               title_tag,
               occurence_threshold=None):
    
    self.logger = logger
    self.DEBUG = DEBUG
    self.occurence_threshold = occurence_threshold
    
    self.config_data = self.logger.config_data
    self._parse_config_data()
    
    self.domain_name = domain
    self.domain_url = 'https://' + domain
    self.archive = archive
    self.urls_source = self.domain_url + url_cycle_source
    self.cycle = cycle_range
    self.urls_tag = urls_tag
    self.data_tag = data_tag
    self.title_tag = title_tag
    self.doc_lengths = []
    
    self.undesirable_tags = ['de', 'pentru', 'ce', 'cand', 'cum', 'cine', 'sa', 'se', 'nu', 'da', 'din', 
                             'care', 'dupa', 'lui', 'despre', 'era', 'dar', 'doua', 'cel', 'unei', 'sau',
                              'este', 'mai', 'fost', '000', 'in']
    
    self.article_sources = []
    
    self.cycle_for_urls()

    self.documents, self.labels = self.get_labeled_documents()
    
    self.process_labels()
    
  
  def _parse_config_data(self):
    if self.occurence_threshold is None:
      self.occurence_threshold = self.config_data['OCCURENCE_THRESHOLD']
    return
  
  def validate_url(self, url):
    request = requests.get(url)
    if request.status_code == 200:
#        if self.DEBUG: self.logger.P('Website exists')
        return True
    else:
        self.logger.P('URL {} does not exist'.format(url))
        return False
  
  #find links on page_url in urls_tag
  def get_news_urls(self, page_url, urls_tag):
    #get links for articles
    site_request = requests.get(page_url)
    page_content = site_request.content
    url_soup = BeautifulSoup(page_content, 'html.parser')
      
    list_of_articles = url_soup.find_all(urls_tag[0], class_=urls_tag[1])
    if not list_of_articles:
      self.logger.P("urls_tag [{}] is not yielding links".format(urls_tag))
    
    list_of_links = []
    
    for i in list_of_articles:
      if urls_tag[0] != 'a':
        link = i.find('a')['href']
      else:
        link = i['href']
      
      if(link[:4] == 'http'):
        list_of_links.append(link)
      
      else:
        if(self.validate_url(self.domain_url + link) == True):
          list_of_links.append(self.domain_url + link)
    
    return list_of_links
  
  def cycle_for_urls(self):
    self.article_sources = []
    if(len(self.archive) == 0):
      for i in self.cycle:
        url_cycle = self.urls_source + str(i)
        self.logger.P('Getting urls for news from {}'.format(url_cycle))
        articles_on_page = self.get_news_urls(url_cycle, self.urls_tag)
#        self.validate_url(url_cycle)
        self.article_sources.append(articles_on_page)
        self.logger.P('Found {} stories'.format(len(articles_on_page)))
    else:
      archive_url = self.domain_url + self.archive[0]
      url_pages = self.get_news_urls(archive_url, self.archive[1])
      for page in url_pages[:50]:
        if self.DEBUG:self.logger.P('Getting urls from {}...'.format(page))
        articles_on_page = self.get_news_urls(page, self.urls_tag) 
        self.article_sources.append(articles_on_page)
        self.logger.P('Found stories on this page: {}'.format(len(articles_on_page)))
      
    self.article_sources = flatten_list(self.article_sources)
    self.logger.P('Total number of stories {}'.format(len(self.article_sources)))
  
  #get text and label from url    
  def get_text_and_label(self, url):
    news_request = requests.get(url)
    if self.DEBUG: self.logger.P('URL {} returned status code {}'.format(url, news_request.status_code))
    news_source = BeautifulSoup(news_request.content, 'html.parser')
    
    #process class tags if empty
    if self.data_tag[1] == '':
      self.data_tag[1] = None
    
    if self.data_tag[3] == '':
      self.data_tag[3] = None
    
    if self.title_tag[1] == '':
      self.title_tag[1] = 0
    
    if self.title_tag[3] == '': 
      self.title_tag[3] = None
      
    #populate document
    news_article = news_source.find(self.data_tag[0], 
                                    class_=self.data_tag[1])
    #check tags
    if news_article is None:
      self.docs_lengths.append(0)
      self.logger.P('Found no <{} class={}> at url {}'.format(self.data_tag[0], self.data_tag[1], url))
      return [], []

    #get text data
    news_text = news_article.find_all(self.data_tag[2], 
                                      class_=self.data_tag[3])
    text = ''
    for i in news_text:
      text = text + i.get_text()
    
    #keep document length
    length_of_doc = nltk.tokenize.word_tokenize(text)    
    self.doc_lengths.append(len(length_of_doc))

    if self.DEBUG:  self.logger.P('Found <{} class={}> tag in the webpage, found {} number of <{} class={}> tag(s) containing {} words'.format(self.data_tag[0], self.data_tag[1], len(news_text) ,self.data_tag[2], self.data_tag[3], len(text.split())))
  
    # populate labels with metadata
    metatags = news_source.find_all('meta',attrs={'name':'keywords'})
    labels = []
    for tag in metatags:
      s = tag.get('content')
      s = self.tokenizer.tokenize(s.lower())
      #remove words from list of undesirable words
      for i in s:
        if i in self.undesirable_tags:
          s.remove(i)

      labels.append(s)      
    
    #get tags from title (if necessary)
    title = news_source.find('meta',attrs={'property':'og:title'})
    title = title.get('content')
    title_tags = self.process_title(title)
    labels.append(title_tags)
    
    #flatten list
    labels = flatten_list(labels)
    
    #remove duplicates
    labels = list(set(labels))
    for i in labels:
      if len(i) < 5:
        print(i)
        labels.remove(i)
#    
    return text, labels

  #get document and label list from all urls in article_source   
  def get_labeled_documents(self):
    documents = []
    labels = []
    meta_data = []
    self.tokenizer = RegexpTokenizer(r'\w+')
    self.logger.P('Getting texts and labels from found articles...')
    for i in tqdm(self.article_sources):
      doc, lbl = self.get_text_and_label(i)
      documents.append(doc)
      labels.append(lbl)
      meta_data.append(i)
      
    df_lengths = pd.DataFrame(columns=['doc_len'])
    df_lengths.doc_len = self.doc_lengths
    df_length_distrib = df_lengths.describe()

    self.logger.P("Distribution of document lengths as extracted on {}: \n {}".format(self.domain_url, df_length_distrib.to_string()))

    self.min_document_length = df_length_distrib.loc['25%']['doc_len']
    self.max_document_length = df_length_distrib.loc['75%']['doc_len']
    
    return documents, labels
  
  def remove_tag_from_labels(self, tag):
    self.logger.P('Removing tag {} from all labels...'.format(tag))
    for i in range(len(self.labels)):
      try:
        self.labels[i].remove(tag)
      except:
        if self.DEBUG:self.logger.P('Tried removing tag {}, not found'.format(tag))
        pass


  #method to return tags from title in url 
  def process_title(self, title):
    title_tags = self.tokenizer.tokenize(title)
    
    #remove string numbers larger than 3000 
    for i in title_tags:
      if(is_number_larger_than_x(i,3000)) or i in self.undesirable_tags:
        title_tags.remove(i)

    return title_tags

  def process_labels(self):
    self.flattened_labels = flatten_list(self.labels)
    self.dict_label_occurence = Counter(self.flattened_labels)
    self.common_labels = []

    if self.DEBUG: self.logger.P('Word frequency in documents before common tag removal:\n {}'.format(self.dict_label_occurence))
    
    #REMOVE COMMON WORDS
    for i in self.dict_label_occurence.keys():
      percentage = self.dict_label_occurence.get(i)/len(self.documents)
      if percentage > self.occurence_threshold:
        self.logger.P('{} word appears in {}% of documents, will be removed from dataset'.format(i, str(percentage * 100)))
        self.common_labels.append(i)


    self.logger.P('Most common tags that will be removed from list of labels:')
    for i in self.common_labels:
      self.remove_tag_from_labels(i)
    
    self.logger.P('Total number of removed labels {}'.format(len(self.common_labels)))  

    #update flattened labels to exclude removed labels
    self.flattened_labels = flatten_list(self.labels)
    self.dict_label_occurence = Counter(self.flattened_labels)

    self.logger.P('Word frequency in documents:\n {}'.format(self.dict_label_occurence))


    self.inv_dict_label_occurence = {}
    for k, count in self.dict_label_occurence.items():
      try:
        self.inv_dict_label_occurence[count].append(k)
      except KeyError:
        self.inv_dict_label_occurence[count] = [k]

    self.dict_label_count = {}
    total_count = 0
    for k, v in self.inv_dict_label_occurence.items():
      length = len(v)
      self.dict_label_count[k] = length
      total_count += k * length

    self.lengths_of_labels = []
    for i in range(len(self.labels)):
      self.lengths_of_labels.append(len(self.labels[i]))

    df_lbl_len = pd.DataFrame(columns=['len'])
    df_lbl_len.len = self.lengths_of_labels
    df_lbl_len_distrib = df_lbl_len.describe()

    self.logger.P('The distribution of lengths of labels for each document: \n {}'.format(df_lbl_len_distrib.to_string()))

    self.logger.P('Length of flattened labels array {} must be equal to added values in dict of word lengths {}'.format(len(self.flattened_labels), total_count))

    self.logger.P('Labels grouped by frequency \n {}'.format(self.dict_label_count))

    df = pd.DataFrame(columns=['labels'])
    df.labels = self.flattened_labels
    df_distrib = df.describe(include='all')

    self.logger.P("Distribution of labels: \n {}".format(df_distrib.to_string()))

    return
  
  def remove_data_row(self, index):
    self.logger.P('Deleting row {}'.format(index))
    del self.documents[index]
    del self.labels[index]
    del self.doc_lengths[index]
    
  def process_texts(self):
    self.logger.P('To remove outliers, will keep the documents with lengths between 25 and 75 percentiles of document lengths...')
    self.logger.P('Total number of documents before cleaning {}'.format(len(self.documents)))
    
    i = 0
    n = len(self.labels)
    while i < n:
      if self.doc_lengths[i] < self.min_document_length or self.doc_lengths[i] > self.max_document_length or len(self.labels[i]) < 2:
        self.remove_data_row(i)
        n = n - 1
      else:
        i = i + 1

    assert(len(self.documents) == len(self.labels)) 

    self.logger.P('cleaning the data to only include documents of length in range {} {}'.format(self.min_document_length, self.max_document_length))
    self.logger.P('Total number of documents left {}'.format(len(self.documents)))

      
if __name__ == '__main__':
  logger = Logger(lib_name='DOC-COLLECTOR', 
                config_file='./tagger/crawler/config_crawler.txt')

  Digi24 = Spider(logger, False, 'digi24.ro', [] ,'/stiri/actualitate?p=', range(2,4) , ('h4','article-title'), ['article', 'article-story', 'p',''], ['div','col-8 col-md-9 col-sm-12','h1',''])
  
#  Hotnews = Spider(logger, False, 'hotnews.ro', ['/arhiva/2019', ('td', 'calendarDayEnabled')], '', 0, ('div','result_item'), ['div','articol_render','div',''])
  
