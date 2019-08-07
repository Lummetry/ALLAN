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
               include_title=False,
               occurence_threshold=None,
               min_document_length=None,
               min_label_length=None):
    
    self.logger = logger
    self.DEBUG = DEBUG
    self.occurence_threshold = occurence_threshold
    self.min_document_length = min_document_length
    self.min_label_length = min_label_length
    
    self.config_data = self.logger.config_data
    self._parse_config_data()
    
    self.domain_name = domain
    self.domain_url = 'https://' + domain
    self.archive = archive
    self.urls_source = self.domain_url + url_cycle_source
    self.cycle = cycle_range
    self.urls_tag = urls_tag
    self.data_tag = data_tag
    self.include_title = include_title
    self.doc_lengths = []
    
    self.undesirable_tags = ['de', 'pentru', 'ce', 'cand', 'cum', 'cine', 'sa', 'se', 'nu', 'da', 'din', 
                             'care', 'dupa', 'lui', 'despre', 'era', 'dar', 'doua', 'cel', 'unei', 'sau',
                              'este', 'mai', 'fost', '000', 'in', 'decat', 'incat', 'cele', 'unui', 'nici'
                              'mulți', 'înainte', 'unor']

    
    self.article_sources = []
    
    self.cycle_for_urls()
    
    self.documents, self.labels = self.get_labeled_documents()
      
    self.process_labels()
    
#    print(list(zip(self.documents, self.labels)))
  
  def _parse_config_data(self):
    if self.occurence_threshold is None:
      self.occurence_threshold = self.config_data['OCCURENCE_THRESHOLD']
    if self.min_label_length is None:
      self.min_label_length = self.config_data['MINIMUM_LABEL_LENGTH']
    if self.min_document_length is None:
      self.min_document_length = self.config_data['MINIMUM_DOCUMENT_LENGTH']
    return
  
  def validate_url(self, url):
    request = requests.get(url)
    if request.status_code == 200:
        return True
    else:
        self.logger.P('URL {} does not exist'.format(url))
        return False
  
  def check_url_same_domain(self, url):
    if self.domain_name in url:
      return True
    else:
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
        links = i.find_all('a')
        for j in links:
          link = j['href']
          if self.check_url_same_domain(link):
            list_of_links.append(link)
          else:
            if(self.validate_url(self.domain_url + link) == True):
              list_of_links.append(self.domain_url + link)
      else:
        link = i.find('a')
    
    return list_of_links
  
  def cycle_for_urls(self):
    self.article_sources = []
    #NO ARCHIVE, CYCLE THROUGH PAGES LIKE THE 'o's in google
    if(len(self.archive) == 0):
      for i in self.cycle:
        url_cycle = self.urls_source + str(i)
        articles_on_page = self.get_news_urls(url_cycle, self.urls_tag)
        self.article_sources.append(articles_on_page)
        self.logger.P('Found {} stories on {}'.format(len(articles_on_page), url_cycle))
    #go throgh archive and collect a list of pages with article urls     
    else:
      archive_url = self.domain_url + self.archive[0]
      url_pages = self.get_news_urls(archive_url, self.archive[1])
      for page in url_pages[:1]:
        articles_on_page = self.get_news_urls(page, self.urls_tag) 
        self.article_sources.append(articles_on_page)
        self.logger.P('Found {} stories on {}'.format(len(articles_on_page), page))
      
    self.article_sources = flatten_list(self.article_sources)
    self.logger.P('Total number of stories {}'.format(len(self.article_sources)))
  
  #get text and label from url    
  def get_text_and_label(self, url):
    self.logger.start_timer("request_url")
    news_request = requests.get(url)
    self.logger.end_timer("request_url")
    if self.DEBUG: self.logger.P('URL {} returned status code {}'.format(url, news_request.status_code))
    news_source = BeautifulSoup(news_request.content, 'html.parser')
    
    #process class tags if empty
    if self.data_tag[1] == '':
      self.data_tag[1] = None
    
    if self.data_tag[3] == '':
      self.data_tag[3] = None
      
    #populate document
    news_article = news_source.find(self.data_tag[0], 
                                    class_=self.data_tag[1])
    #check tags
    if news_article is None:
      self.doc_lengths.append(0)
      if self.DEBUG: self.logger.P('Found no <{} class={}> at url {}'.format(self.data_tag[0], self.data_tag[1], url))
      return [], []

    #get decoded text data
    news_text = news_article.find_all(self.data_tag[2], 
                                      class_=self.data_tag[3])
    text = ''
    for i in news_text:
      text = text + i.get_text().encode('utf-8').decode() + ' '
    text = text[:-1]
    
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
      #only add words not in undesirable list and of appropriate length
      for i in s:
        if i not in self.undesirable_tags and len(i) > self.min_label_length:
          labels.append(i)
    
    #get tags from title (if necessary)
    title = news_source.find('meta',attrs={'property':'og:title'})
    title = title.get('content')
    title_tags = self.process_title(title.lower())
    if self.include_title:
      for i in title_tags:
        labels.append(i)

    
    #remove duplicates
    labels = list(set(labels))
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

    if self.DEBUG:self.logger.P("Distribution of document lengths as extracted on {}: \n {}".format(self.domain_url, df_length_distrib.to_string()))

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


  #method to return tags from title 
  def process_title(self, title):
    title_tags = []
    tags = self.tokenizer.tokenize(title)
    #remove string numbers larger than 3000 
    for i in tags:
      if len(i) > self.min_label_length:
        if not ((is_number_larger_than_x(i,3000)) or i in self.undesirable_tags):
          title_tags.append(i)

    return title_tags

  def process_labels(self):
    self.flattened_labels = flatten_list(self.labels)
    self.dict_label_occurence = Counter(self.flattened_labels)
    self.common_labels = []

    if self.DEBUG: 
      self.logger.P('Word frequency in documents before common tag removal:\n {}'.format(self.dict_label_occurence))
      self.logger.P('label information before removing common labels and removing document lengths')
      self.label_information()

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

    self.process_texts()
    
    self.logger.P('---------- label information following cleaning ----------', noprefix=True)
    self.label_information()
    self.logger.P('---------- END label information following cleaning ----------', noprefix=True)

  
    return
  
  def label_information(self):
    #update flattened labels
    self.flattened_labels = flatten_list(self.labels)
    #update label occurence counter
    self.dict_label_occurence = Counter(self.flattened_labels)
    
    self.logger.P('Word frequency in labels:\n {}'.format(self.dict_label_occurence))

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
      self.dict_label_count[k] = [length, str((k/len(self.documents))*100)[:6] + '%']
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
    del self.article_sources[index]
    
  def process_texts(self):
    self.min_document_length = 25
    
    self.logger.P('To remove outliers, will keep the documents with lengths between {} and {}...'.format(self.min_document_length, self.max_document_length))
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

#CLASS CONTAINING MULTIPLE SPIDERS

class Dataset(object):
  def __init__(self, logger, DEBUG, DEBUG_SPIDERS, list_of_params, 
               occurence_threshold=None,
               min_label_length=None,
               min_document_length=None):
    
    self.logger = logger
    self.DEBUG = DEBUG
    
    self.occurence_threshold = occurence_threshold
    self.min_document_length = min_document_length
    self.min_label_length = min_label_length
    
    self.list_of_spiders = []
    
    self.config_data = self.logger.config_data
    self._parse_config_data()

    self.list_of_params = list_of_params
    
    self.start_crawl()
    
    self.label_information()
    
    self.write_data()
    
  def start_crawl(self):
    self.documents = []
    self.labels = []
    self.document_lengths = []
    self.metadata = []
    self.logger.P('-------------------------- START CRAWL --------------------------', noprefix=True)
    for i in self.list_of_params:
      self.logger.P('Started crawling {}'.format(i[0] + i[2]))
      self.logger.P('---------------- LOGGER ON SPIDER {} ----------------'.format((i[0] + i[2])), noprefix=True)
      s = Spider(self.logger, self.DEBUG, i[0], i[1], i[2], i[3], i[4], i[5], i[6], self.occurence_threshold, self.min_document_length, self.min_label_length)
      self.logger.P('---------------- END LOGGER ON SPIDER {} ----------------'.format((i[0] + i[2])), noprefix=True)

      self.list_of_spiders.append(s)
      self.documents.append(s.documents)
      self.labels.append(s.labels)
      self.document_lengths.append(s.doc_lengths)
      self.metadata.append(s.article_sources)
      self.logger.P('Collected {} documents from {}'.format(len(s.documents),i[0] + i[2]))
      
    self.logger.P('-------------------------- END CRAWL --------------------------', noprefix=True)
    self.logger.P('Data collection complete.')
    self.documents = flatten_list(self.documents)
    self.labels = flatten_list(self.labels)
    self.document_lengths = flatten_list(self.document_lengths)
    self.metadata = flatten_list(self.metadata)

  def _parse_config_data(self):
    if self.occurence_threshold is None:
      self.occurence_threshold = self.config_data['OCCURENCE_THRESHOLD']
    if self.min_label_length is None:
      self.min_label_length = self.config_data['MINIMUM_LABEL_LENGTH']
    if self.min_document_length is None:
      self.min_document_length = self.config_data['MINIMUM_DOCUMENT_LENGTH']
      
    return
  
  def label_information(self):
    #update flattened labels
    self.logger.P('\n ----------------------------- Label information on dataset ----------------------------', noprefix=True)
    self.logger.P('Displaying information regarding the labels of the whole datatset...')
    self.flattened_labels = flatten_list(self.labels)
    #update label occurence counter
    self.dict_label_occurence = Counter(self.flattened_labels)
    
    self.logger.P('Word frequency in labels:\n {}'.format(self.dict_label_occurence))

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
      self.dict_label_count[k] = [length]
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
    
    df_lengths = pd.DataFrame(columns=['doc_len'])
    df_lengths.doc_len = self.document_lengths
    df_length_distrib = df_lengths.describe()

    self.logger.P("Distribution of document lengths {}".format(df_length_distrib.to_string()))

    df = pd.DataFrame(columns=['labels'])
    df.labels = self.flattened_labels
    df_distrib = df.describe(include='all')

    self.logger.P("Distribution of labels: \n {}".format(df_distrib.to_string()))

    self.logger.P('----------------------------- END Label information on dataset -----------------------------', noprefix=True)

    return
  
  def write_data(self):
    for i in range(len(self.documents)):
      with open('./tagger/crawler/Texts/Text_%s.txt' % i, 'w') as f_doc:
        f_doc.write(self.documents[i])

      with open('./tagger/crawler/Labels/Label_%s.txt' % i, 'w') as f_label:
        for j in self.labels[i]:
          f_label.write(j + '\n')
        
      with open('./tagger/crawler/Meta/Meta_%s.txt' % i, 'w') as f_meta:
        f_meta.write(self.metadata[i] + '\n')
        f_meta.write('Document_length: ' + str(self.document_lengths[i]))
      
      f_doc.close()
      f_label.close()
      f_meta.close()
      
if __name__ == '__main__':
  logger = Logger(lib_name='DOC-COLLECTOR', 
                config_file='./tagger/crawler/config_crawler.txt', TF_KERAS=False)

#  stiri_pe_surse = Spider(logger, True, 'stiripesurse.ro', ['/arhiva/', ('section', 'left-container column ld-three-fourths md-two-thirds sd-one-full')], '', 0, ('section', 'main container_16'), ['article', 'post article-single', 'p', ''], [])
  
  list_of_parameters = [['digi24.ro', [] ,'/stiri/actualitate?p=', range(1,2) , ('h4','article-title'), ['article', 'article-story', 'p',''], False],
                        ['hotnews.ro', ['/arhiva/2019', ('td', 'calendarDayEnabled')], '', 0, ('div','result_item'), ['div','articol_render','div',''], False],
                        ['digi24.ro', [] ,'/stiri/actualitate?p=', range(1,2) , ('h4','article-title'), ['article', 'article-story', 'p',''], False],
                        ['digi24.ro', [] ,'/stiri/economie?p=', range(1,2) , ('h4','article-title'), ['article', 'article-story', 'p',''], False]]
  
  crawled_data = Dataset(logger, False, False, list_of_parameters)