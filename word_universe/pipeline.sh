#!/bin/bash


# ./pipeline.sh -e tf_20 -p base_folder -a app_folder -c corpus_folder -d false -m full_path_to_transfer_model 

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=Mac;;
    CYGWIN*)    machine=Cygwin;;
    MINGW*)     machine=MinGw;;
    *)          machine="UNKNOWN:${unameOut}"
esac

ENVIRONMENT=tf_20
link=https://dumps.wikimedia.org/rowiki/20200120/rowiki-20200120-pages-articles.xml.bz2
path=Dropbox
app_path=_allan_data/_rowiki_dump
corpus_folder=corpus
download=true
model=

while getopts e:l:p:c:a:d:m: option
do
case "${option}"
in
e) ENVIRONMENT=${OPTARG};;
l) link=${OPTARG};;
p) path=${OPTARG};;
c) corpus_folder=${OPTARG};;
a) app_path=${OPTARG};;
d) download=${OPTARG};;
m) model=${OPTARG};;
esac
done

source ~/anaconda3/etc/profile.d/conda.sh
conda activate $ENVIRONMENT
echo "[INFO] Environment $ENVIRONMENT activated"

if [ "$download" = true ] ; then
	xml_bz2_name=rowiki-pages-article.xml.bz2
	xml_name=$(echo "$xml_bz2_name" | sed "s/\(.*\).\{4\}/\1/")

	full_path="$path/$app_path/_data/$xml_bz2_name"
	echo "[INFO] Downloading rowiki dump from $link ..."
	curl -o "$full_path" $link
	echo "[INFO]  Download completed ($full_path)!"

	cd "$path/$app_path/_data"
	echo "[INFO] Unarchiving $xml_bz2_name"
	bzip2 -d $xml_bz2_name
	echo "[INFO]  Unarchive completed!"

	echo "[INFO] Extracting data from xml ..."
	python wikiextractor-master/WikiExtractor.py -o $corpus_folder --quiet $xml_name
	echo "[INFO]  Extraction completed!"

	echo "[INFO] Deleting xml file ..."
	rm -f $xml_name
	echo "[INFO]  Deletion completed!"
fi

echo "[INFO] Checking necessary utilitaries ..."
if [ "$machine" = Mac ]; then
	brew list jq || brew install jq
else
	dpkg -s jq || sudo apt-get install jq
fi
echo "[INFO]  Utilitaries checked!"

cd "$SCRIPTPATH"
cat config.txt | jq --arg a "$path" '.BASE_FOLDER = $a' | jq --arg a "$app_path" '.APP_FOLDER = $a' | jq --arg a "$corpus_folder" '.CORPUS_FOLDER = $a' | jq --arg a "$model" '.TRANSFER_MODEL_REAL_PATH = $a' > config_tmp.txt && rm config.txt && mv config_tmp.txt config.txt
cd ..
echo "$(pwd)"
export PYTHONPATH=$PYTHONPATH:"$(pwd)"
echo "[INFO] Merging corpus and creating word2vec..."
python word_universe/main.py
echo "[INFO] Process completed!"
