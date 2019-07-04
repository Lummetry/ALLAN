import glob, os
import requests
import json

folder = "F:\\LummetryAI.Dropbox\\Lummetry.AI Dropbox\\DATA\\_doc_ro_chatbot_data\\00_Corpus\\00_mihai_work\\20190226_Production_selection_v0_3\\labels"


def get_txt_files_names(folder):
    os.chdir(folder)
    files = []
    for file in glob.glob("*.txt"):
        files.append(file)
    return files

def read_file(file, folder):
    input_file = open(os.path.join(folder, file), 'r', encoding="utf8")
    file_contents = input_file.read()
    input_file.close()
    word_list = file_contents.split()
    return word_list


files = get_txt_files_names(folder)
lst_all_words = []
for f in files:
    r = read_file(f, folder)
    lst_all_words.extend(r)
    print(len(r))
    print(f)
print(len(lst_all_words))
print(len(set(lst_all_words)))
for wd in set(lst_all_words):
    print(wd)
    rm = requests.post(url="http://127.0.0.1:8000/api_create_label/", data={'label':wd})

    dt = rm.json()
    print(dt)