# News Article Crawler
This module is intended to provide an interface for extracting the news articles residing on any news website.

The data it extracts is of the form: texts -> labels. Where the texts are the content of the news article and the labels are the the meta tags of the news's article's page.

Furthermore, the labels are cleaned based on a couple of heuristics: 
1. A minimum label length - labels shorterr than this parameters are removed.
2. Occurence threshold    - labels that occur in a higher percentage of all documents than this parameters are removed.
3. An include title flag  - boolean indicating whether the tokenized title should be added to the labels list.

This crawler has two modes:
  1. Archive mode - where if the archive parameter is not '', the crawler checks a single archive page for
                 a list of pages with links to articles. It cycles through these pages and populates 
                 a list of urls to articles. It then cycles through the list of urls to articles and
                 populates a dataset with texts and labels.

  2. Url_cycle mode - where if the archive parameter is '', the crawler goes through the url_cylce_source 
                   pages in the cycle_range(e.g. '/stiri/actualitate?p=1' to '/stiri/actualitate?p=20')
                   and generates links to articles. It goes through the list of articles and populates 
                   a dataset containing texts and labels.
# How to use the crawler

1. Change the parameters in the config file(config_crawler.txt) to fit your needs.

2. Add the details of the websites you desire to crawl to the list of parameters in __main__ like so:

    The list of parameters is: 
    a list of lists, each list containing the required parameters for a spider to crawl:
    
    * archive - a string pointing to a subpage of the website, where an archive resides
               the subpage contains a list of links where each link points to a page with a 
               list of links to single articles

    * url_cycle_source - a subpage such as '/stiri/actualitate?p=' where after 'p=' a number
                        could be added so that the crawler can cylce through pages containing
                        links to articles

    * cycle_range - the range of numbers added to url_cycle_source

    * urls_tag - a tuple containing the html tag and class (tag, class) where the urls to articles
                can be found on the url_cycle_source pages

    * data_tag - a list containing two tuples, each tuple contains an html tag and class (tag, class)
                where the article information (actual article story) is found. The second tuple 
                is nested in the html code of the page inside of the first tuple.

    * include_title - flag indicating whether the crawler should save the tokenized title in the labels
                    of each document. If true, the tags generated from the title are stored in a 
                    separate list.

    See Spider object docstring for further information.

3. Running the script and as such will display valuable information about how the data collected. 

4. Once happy with the data, add write_to_file() method to the _main_ and the dataset will be written to disk.

# Findings

To keep the distribution of labels valuable to a document tagger, make sure that the websites crawled metadata keywords are good and human annotated.
Some websites split their titles to generate the metadata keywords, this makes for noisy labels.
