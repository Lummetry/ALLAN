import os
from chatbot.word_universe.doc_utils import DocUtils
from tqdm import tqdm

def parse_and_merge_files(path, output):
    w = open(output, "wt")

    pbar = tqdm(os.walk(path))
    for root, subdirs, files in pbar:
        for filename in files:
            if filename.startswith('.'): continue
            pbar.set_description("Folder: {}, Processing file: {}".format(root[-20:], filename[-20:]))

            with open(os.path.join(root,filename), "rt") as f:
                for line in f.readlines():

                    if line.startswith('<doc') or \
                        line.startswith('</doc') or \
                        line.startswith('[[') or \
                        line.startswith('!style'):
                        continue

                    #line_mod = DocUtils.strip_html_tags(line)
                    line_mod = DocUtils.prepare_for_tokenization(line)

                    w.write(line_mod)

    w.close()


if __name__ == "__main__":
  path = '/Users/laurentiupiciu/Lummetry.AI Dropbox/DATA/_allan_data/_rowiki_dump/_data/corpus/'
  output = '/Users/laurentiupiciu/Lummetry.AI Dropbox/DATA/_allan_data/_rowiki_dump/_data/20200113_rowiki'
  
  
  parse_and_merge_files(path, output)