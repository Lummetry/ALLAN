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
  """
  Function to check whether a number as a string is larger than another number
  
  Arguments:number - number as a string
            x - int number
  
  Returns true if the number is larger,
  """
  try:
    if int(number) > x:
      return True
    else:
      return False
  except ValueError:
    return False

class Spider(object):
  """
  Class that performs crawling on a single website. Generates texts, labels pairs, where the texts are
  news articles and the labels are drawn from the meta tags of the website, and, on request, the labels 
  can also contain labels from the tokenized title of the article
  
  This crawler has two modes:
    archive mode - where if the archive parameter is not '', the crawler checks a single archive page for
                   a list of pages with links to articles. It cycles through these pages and populates 
                   a list of urls to articles. It then cycles through the list of urls to articles and
                   populates a dataset with texts and labels.
                   
    url_cycle mode - where if the archive parameter is '', the crawler goes through the url_cylce_source 
                     pages in the cycle_range(e.g. '/stiri/actualitate?p=1' to '/stiri/actualitate?p=20')
                     and generates links to articles. It goes through the list of articles and populates 
                     a dataset containing texts and labels.
  
  Arguments: logger
  
             DEBUG - boolean, if true the logger is verbose
             
             archive - a string pointing to a subpage of the website, where an archive resides
                       the subpage contains a list of links where each link points to a page with a 
                       list of links to single articles
                       
             url_cycle_source - a subpage such as '/stiri/actualitate?p=' where after 'p=' a number
                                could be added so that the crawler can cylce through pages containing
                                links to articles
            
             cycle_range - the range of numbers added to url_cycle_source
             
             urls_tag - a tuple containing the html tag and class (tag, class) where the urls to articles
                        can be found on the url_cycle_source pages
                        
             data_tag - a list containing two tuples, each tuple contains an html tag and class (tag, class)
                        where the article information (actual article story) is found. The second tuple 
                        is nested in the html code of the page inside of the first tuple.
            
            include_title - flag indicating whether the crawler should save the tokenized title in the labels
                            of each document. If true, the tags generated from the title are stored in a 
                            separate list.
                        
  """
  def __init__(self, logger, DEBUG,
               domain,
               archive,
               url_cycle_source,
               cycle_range,
               urls_tag,
               data_tag,
               include_title=False):
    """
    Constructor that loads config data, populates list of urls, generates documents and labels.
    
    The config data contains: base and app folder to dump generated data
                              MINIMUM_DOCUMENT_LENGTH - int, smallest length of tokenized document allowed, 
                                                        the smaller ones are discarded
                              OCCURENCE_THRESHOLD - Float between  0 - 1
                                                    percentage over which tags are removed from dataset
                                                    if a tag occurs in enough documents, it is discarded
                              
                              MINIMUM_LABEL_LENGTH - int, tags of shorter length are discarded
                              
    """
    self.logger = logger
    self.DEBUG = DEBUG
    
    self.include_title = include_title
    self.config_data = self.logger.config_data
    
    self.domain_name = domain
    self.domain_url = 'https://' + domain
    self.archive = archive
    self.urls_source = self.domain_url + url_cycle_source
    self.cycle = cycle_range
    self.urls_tag = urls_tag
    self.data_tag = data_tag
    self.doc_lengths = []
    
    self.article_sources = []
    
    self.cycle_for_urls()
    
    self.documents, self.labels, self.title_tags = self.get_labeled_documents()
      
    
  def validate_url(self, url):
    """
    Function that checks whether a URL returns 200.
    
    Arguments - url - String pointing to a web page
    
    Returns True/False
    """
    request = requests.get(url)
    if request.status_code == 200:
        return True
    else:
        self.logger.P('URL {} does not exist'.format(url))
        return False
  
  def check_url_same_domain(self, url):
    """
    Function that checks whether a URL is in the domain of the spider
    
    Arguments - url - String pointing to a web page
    
    Returns True/False
    """
    if self.domain_name in url:
      return True
    else:
      return False
  
  #find links on page_url in urls_tag
  def get_news_urls(self, page_url, urls_tag):
    """
    Finds urls to news articles on a single page.
    
    Arguments: page_url - the page where links to news articles reside
               urls_tag - html tag containing the urls to news articles
    
    Returns: a list of links to news articles
    """
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
    """
    Function that cycles the domain in either the archive of the url_cycle_source for article urls.
    
    Populates self.article_sources which the crawler uses to get documents and labels.
    
    If in archive mode, it collects all the news urls from the archive
    
    If in non archive mode, it collects all news urls in cycle_range.
    
    No returns, changes article_sources list of Spider object
    """
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
      for page in url_pages:
        articles_on_page = self.get_news_urls(page, self.urls_tag) 
        self.article_sources.append(articles_on_page)
        self.logger.P('Found {} stories on {}'.format(len(articles_on_page), page))
      
    self.article_sources = flatten_list(self.article_sources)
    self.logger.P('Total number of stories {}'.format(len(self.article_sources)))
  
  #get text and label from url    
  def get_text_and_label(self, url):
    """
    Atomic function that finds the text, labels and title labels of a single news article.
    
    Arguments: url to a page containing a news article
    
    Returns: text - string containing the text of the news article, as found by the data_tag
             labels - a list of labels drawn from the meta tags of the page
             title_tags - a list of labels drawn from the tokenized title of the page
    """
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
      if self.DEBUG: self.logger.P('Found no <{} class={}> at url {}'.format(self.data_tag[0], self.data_tag[1], url))
      return [], [], []

    #get decoded text data
    news_text = news_article.find_all(self.data_tag[2], 
                                      class_=self.data_tag[3])
    text = ''
    for i in news_text:
      text = text + i.get_text() + ' '
    text = text[:-1]
    
    if self.DEBUG:  self.logger.P('Found <{} class={}> tag in the webpage, found {} number of <{} class={}> tag(s) containing {} words'.format(self.data_tag[0], self.data_tag[1], len(news_text) ,self.data_tag[2], self.data_tag[3], len(text.split())))
  
    # populate labels with metadata
    metatags = news_source.find_all('meta',attrs={'name':'keywords'})
    labels = []
    for tag in metatags:
      s = tag.get('content')
      s = self.tokenizer.tokenize(s.lower())
      #only add words not in undesirable list and of appropriate length
      for i in s:
        labels.append(i)
    
    #get tags from title (if necessary)
    title = news_source.find('meta',attrs={'property':'og:title'})
    if title is None:
      self.logger.P('[ERROR]Meta tag containing title not found...')
    title = title.get('content')
    
    title_tags = []
    
    split_title = self.tokenizer.tokenize(title.lower())
    for i in split_title:
      title_tags.append(i)
    
    if self.include_title:
      for i in title_tags:
        labels.append(i)
    
    #remove duplicates
    labels = list(set(labels))
  
    return text, labels, title_tags

  #get document and label list from all urls in article_source   
  def get_labeled_documents(self):
    """
    Iterate through the list of news articles and populate a list of documents, a list of labels 
    and a list of title_labels.
    
    Returns: documents - a list of strings where each item is the text of an article
             labels - a list of lists of labels for each document
             title_labels - a list of lists of labels for each document- as drawn from titles.
                            is empty if include_titles is False.
    """
    documents = []
    labels = []
    title_labels = []
    meta_data = []
    self.tokenizer = RegexpTokenizer(r'\w+')
    self.logger.P('Getting texts and labels from found articles...')
    
    for i in tqdm(self.article_sources):
      doc, lbl, title_lbl = self.get_text_and_label(i)
      if len(doc) > 0 and len(lbl) > 0:
        documents.append(doc)
        labels.append(lbl)
        meta_data.append(i)
        #keep document lengths
        document_length = len(self.tokenizer.tokenize(doc))
        self.doc_lengths.append(document_length)
      
        if self.include_title == False:
          title_labels.append(title_lbl)
      
    df_lengths = pd.DataFrame(columns=['doc_len'])
    df_lengths.doc_len = self.doc_lengths
    df_length_distrib = df_lengths.describe()
    self.max_document_length = df_length_distrib.loc['75%']['doc_len']
    
    if self.DEBUG:self.logger.P("Distribution of document lengths as extracted on {}: \n {}".format(self.domain_url, df_length_distrib.to_string()))
    
    return documents, labels, title_labels
  
class Crawled_Dataset(object):
  """
  Dataset object, the shared result of multiple crawlers
  
  Arguments: logger
             DEBUG - boolean, if true the logger is verbose
             spider_params - a list of lists, each list containing the required parameters for a spider 
                             to crawl: domain, archive, url_cycle_source, cycle_range, urls_tag, data_tag
                             See Spider object documentation for further information.
             include_titles - boolean, will add tags from tokenized title if true
             
             The following parameters are meant to be loaded from the config file, if values are specified
             upon declaring the crawled_dataset object, the config file data will be overridden and the 
             class will use the values assigned by the user in the declaration:
             
             occurence_threshold - Float between  0 - 1
                                    percentage over which tags are removed from dataset
                                    if a tag occurs in enough documents, it is discarded
             min_document_length - int, smallest length of tokenized document allowed, 
                                        the smaller ones are discarded              
             min_label_length - int, tags of shorter length are discarded
  """
  def __init__(self, logger, DEBUG, 
               spider_params,
               include_titles=False,
               occurence_threshold=None,
               min_document_length=None,
               min_label_length=None):
    """
    Constructor that parses config data, performs crawl for each spider, processes labels and texts
                     and writes them to disk
                              
    """
    self.logger = logger
    self.DEBUG = DEBUG
    
    self.spider_params = spider_params
    self.include_titles = include_titles
        
    self.occurence_threshold = occurence_threshold
    self.min_document_length = min_document_length
    self.min_label_length = min_label_length
    
    self.config_data = self.logger.config_data
    self._parse_config_data()
     
    #get data from websites
    self.start_crawl()
    
    self.logger.P('----------------------- LABEL INFORMATION ON RAW DATASET -----------------------', noprefix=True)
    self.label_information(self.labels)
    self.logger.P('----------------------- END LABEL INFORMATION ON RAW DATASET -----------------------', noprefix=True)
    
    #process data 
    self.process_labels()
    self.document_information()
    self.process_texts()
    
    
    self.logger.P('----------------------- LABEL INFORMATION ON PROCESSED DATASET -----------------------', noprefix=True)
    self.label_information(self.labels)
    self.logger.P('----------------------- END LABEL INFORMATION ON  DATASET -----------------------', noprefix=True)
    
    self.write_data()

  def _parse_config_data(self):
    """
    Function that parses the config data, populates the parameters with the values from the config file
    if their values are None.
    """
    if self.occurence_threshold is None:
      self.occurence_threshold = self.config_data['OCCURENCE_THRESHOLD']
      self.logger.P('Initialized occurence threshold with {}'.format(self.occurence_threshold))
    if self.min_label_length is None:
      self.min_label_length = self.config_data['MINIMUM_LABEL_LENGTH']
      self.logger.P('Initialized min label length with {}'.format(self.min_label_length))
    if self.min_document_length is None:
      self.min_document_length = self.config_data['MINIMUM_DOCUMENT_LENGTH']
      self.logger.P('Initialized min doc length with {}'.format(self.min_document_length))

    return
  
  def start_crawl(self):
    """
    Performs crawl for all spiders.
    
    The number of spiders is the length of the spider_params list.
    For each spider, it initializes the spider which calls the constructor of the Spider object.
    Once the object has completed initialization(meaning the crawl is ready and the data is gathered)
    The documents, labels, title_labels, metadata are all added to the dataset.
    Metadata contains information about the url of an article.
    """
    self.spider_list = []
    self.documents = []
    self.labels = []
    self.title_tags = []
    self.document_lengths = []
    self.metadata = []
    
    self.logger.P('-------------------------- START CRAWL --------------------------', noprefix=True)
    for i in self.spider_params:
      self.logger.P('Started crawling {}'.format(i[0] + i[2]))
      self.logger.P('---------------- LOGGER ON SPIDER {} ----------------'.format((i[0] + i[2])), noprefix=True)
      s = Spider(self.logger, self.DEBUG, i[0], i[1], i[2], i[3], i[4], i[5], self.include_titles)
      self.logger.P('---------------- END LOGGER ON SPIDER {} ----------------'.format((i[0] + i[2])), noprefix=True)

      self.spider_list.append(s)
      self.documents.append(s.documents)
      self.labels.append(s.labels)
      self.document_lengths.append(s.doc_lengths)
      self.metadata.append(s.article_sources)
      self.title_tags.append(s.title_tags)
    
      
      self.logger.P('Collected {} documents from {}'.format(len(s.documents),i[0] + i[2]))
      
    self.logger.P('-------------------------- END CRAWL --------------------------', noprefix=True)
    self.logger.P('Data collection complete.')
    self.documents = flatten_list(self.documents)
    self.labels = flatten_list(self.labels)
    self.metadata = flatten_list(self.metadata)
    self.document_lengths = flatten_list(self.document_lengths)
          
    self.logger.P('Got {} documents, with {} labels'.format(len(self.documents), len(self.labels)))
    
    if len(self.title_tags) > 0:
      self.title_tags = flatten_list(self.title_tags)
  
  def document_information(self):
    """
    FunctiOn that display the distribution of document lengths in self.documents
    """
    df_lengths = pd.DataFrame(columns=['doc_len'])
    df_lengths.doc_len = self.document_lengths
    df_length_distrib = df_lengths.describe()

    self.logger.P("Distribution of document lengths: \n {}".format(df_length_distrib.to_string()))
    
    self.max_document_length = df_length_distrib.loc['75%']['doc_len']
  
    return 
    
  def label_information(self, labels):
    """
    Function that displays information on a list of lists of labels.
    
    Arguments: labels - list of lists of labels
    
    No returns, displays the distribution of lengths of labels, the labels grouped by frequency and 
                         the distribution of labels.
    """
    #update flattened labels
    flattened_labels = flatten_list(labels)
    #update label occurence counter
    dict_label_occurence = Counter(flattened_labels)
    
    self.logger.P('Word frequency in labels:\n {}'.format(dict_label_occurence))

    inv_dict_label_occurence = {}
    for k, count in dict_label_occurence.items():
      try:
        inv_dict_label_occurence[count].append(k)
      except KeyError:
        inv_dict_label_occurence[count] = [k]

    dict_label_count = {}
    total_count = 0
    for k, v in inv_dict_label_occurence.items():
      length = len(v)
      dict_label_count[k] = [length]
      total_count += k * length

    lengths_of_labels = []
    for i in range(len(labels)):
      lengths_of_labels.append(len(labels[i]))

    df_lbl_len = pd.DataFrame(columns=['len'])
    df_lbl_len.len = lengths_of_labels
    df_lbl_len_distrib = df_lbl_len.describe()

    self.logger.P('The distribution of lengths of labels for each document: \n {}'.format(df_lbl_len_distrib.to_string()))
    self.logger.P('Length of flattened labels array {} must be equal to added values in dict of word lengths {}'.format(len(flattened_labels), total_count))
    self.logger.P('Labels grouped by frequency \n {}'.format(dict_label_count))

    df = pd.DataFrame(columns=['labels'])
    df.labels = flattened_labels
    df_distrib = df.describe(include='all')

    self.logger.P("Distribution of labels: \n {}".format(df_distrib.to_string()))

    return

  def remove_tag_from_labels(self, tag):
    """
    Function meant to remove a single tag from the self.labels list
    
    Arguments: tag - a label in the self.labels list
    
    No returns, just a change in the self.labels list of the Crawled_Dataset object
    """
    self.logger.P('Removing tag {} from all labels...'.format(tag))
    new_labels = []
    for i in self.labels:
      new_labels.append(list(filter(lambda a: a != tag, i)))
      
    self.labels = new_labels

  def remove_data_row(self, index):
    """
    Function that removes a data row from the dataset.
    
    The dataset is a collection of lists of equal length, so deleting a data row means deleting the entry
    from each of these lists(documents, labels, document_lengths, metadata, title_tags)
    """
    self.logger.P('Deleting row {}'.format(index))
    del self.documents[index]
    del self.labels[index]
    del self.document_lengths[index]
    del self.metadata[index]
    if not self.include_titles:
      del self.title_tags[index]
      
  def generate_dict_label_occ_in_docs(self):
    """
    Function generating a dictionary where keys are labels and values are their occurence
    in the self.labels list of lists.
    """
    self.unqiue_labels  = list(set(flatten_list(self.labels)))
    dictonary = {}
    for i in self.labels:
      for k in self.unqiue_labels:
        if k in i:
          try:
            dictonary[k] += 1
          except:
            dictonary[k] = 1
    
    return dictonary

  def process_labels(self):
    """
    Function that cleans and reduces the number of labels.
    
    Performs reduction of labels based on their occurence percentage, meaning that if a label occurs in 
    more than occurence_threshold percent of data rows, the label will be removed from the dataset.
    
    No returns, only changes in the content of the self.labels list of lists
    """
    self.dict_label_occurence = self.generate_dict_label_occ_in_docs()
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
    
    return
  
  
  def process_texts(self):
    """
    Function that cleans the dataset by removing texts of uninteresting lengths.
    
    After observing the distribuiton of text lengths, only the ones between 25% and 75% are kept.
    This removes outliers and enforces balance in the dataset.
    
    """
    self.min_document_length = 25
    
    self.logger.P('To remove outliers, will keep the documents with lengths between {} and {}...'.format(self.min_document_length, self.max_document_length))
    self.logger.P('Total number of documents before cleaning {}'.format(len(self.documents)))

    i = 0
    n = len(self.labels)
    
    while i < n:
      if self.document_lengths[i] < self.min_document_length or self.document_lengths[i] > self.max_document_length or len(self.labels[i]) < self.min_label_length:
        self.remove_data_row(i)
        n = n - 1
      else:
        i = i + 1

    assert(len(self.documents) == len(self.labels)) 

    self.logger.P('cleaning the data to only include documents of length in range {} {}'.format(self.min_document_length, self.max_document_length))
    self.logger.P('Total number of documents left {}'.format(len(self.documents)))

  def write_data(self):
    """
    Function that writes the data to disk
    
    Texts,
    Labels,
    Metadata,
    Title_tags,
    
    are all stored in the _output folder of the app folder in distinct Folders.
    
    Texts and their labels have the same filenames.
    """
    dir_location = self.logger.GetDropboxDrive()  + '/' + self.logger.config_data['APP_FOLDER']
    for i in range(len(self.documents)):
      with open(dir_location + '/_output/Texts/Text_%s.txt' % i, 'w', encoding='utf-8') as f_doc:
        f_doc.write(self.documents[i])

      f_doc.close()

      with open(dir_location + '/_output/Labels/Text_%s.txt' % i, 'w', encoding='utf-8') as f_label:
        for j in self.labels[i]:
          f_label.write(j + '\n')
      
      f_label.close()
      
      with open(dir_location + '/_output/Meta/Meta_%s.txt' % i, 'w', encoding='utf-8') as f_meta:
        f_meta.write(self.metadata[i] + '\n')
        f_meta.write('Document_length: ' + str(self.document_lengths[i]))
        
      f_meta.close()

      if not self.include_titles:
        with open(dir_location + '/_output/Title_Labels/Title_Labels_%s.txt' % i, 'w', encoding='utf-8') as f_title_labels:
          for t in self.title_tags[i]:
            f_title_labels.write(t + '\n')
        
        f_title_labels.close()
  
     
if __name__ == '__main__':
  logger = Logger(lib_name='DOC-COLLECTOR', 
                  config_file='./tagger/crawler/config_crawler.txt', TF_KERAS=False)

  list_of_parameters = [['digi24.ro', [] ,'/stiri/actualitate?p=', range(1,20), ('h4','article-title'), ['article', 'article-story', 'p','']],
                        ['digi24.ro', [] ,'/stiri/externe?p=', range(1,20), ('h4','article-title'), ['article', 'article-story', 'p','']],
                        ['digi24.ro', [] ,'/stiri/politica?p=', range(1,20), ('h4','article-title'), ['article', 'article-story', 'p','']],
                        ['digi24.ro', [] ,'/stiri/economie?p=', range(1,20), ('h4','article-title'), ['article', 'article-story', 'p','']]]
  
  crawled_data = Crawled_Dataset(logger, False, list_of_parameters, False) 

  a = 0