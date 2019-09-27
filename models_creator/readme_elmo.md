# ELMo Word Embeddings 
In this folder you can find the implementation of the ELMo word embeddings as described in [this paper by Peters et. al](https://arxiv.org/pdf/1802.05365.pdf)

# Structure
In this directory there are 4 files needed to train the language model and generate embeddings.

elmo_main.py
elmo_custom.py
elmo_model.py
config_elmo.txt

elmo_main.py holds a __main__ function where training is initiated and the model is saved to a file. Examples on how to use the visualisation
and embedding generation functions are also present.

elmo_model.py contains the ELMo class where all the data preprocessing, tokenization, batch generation, model definition, output visualization and embedding generation functions are described.

elmo_custom.py contains a couple of custom classes made specially for ELMo, particularly: a Keras custom callback to show detailed metric every 5, and 10 epochs, and a timestep dropout class.

config_elmo.txt is a configuration file where important parameters to the training process can be tweaked. 

# How to use ELMo
In this stage of its development, this class is capable of training, and loading a pretrained model to simply generate embeddings.

For training:

1. instantiate the ELMo class with its needed parameters (the data corpus file and the word2idx mapping)

2. call the .train() method of the ELMo class, this will take the configuration detailed by you in the config file (epochs, batch_size, etc.)

3. make sure to call the .save_elmo_weights() method to save the model you are happy with (it takes a string- the filename as its parameter)


For using a pretrained model:

1. instantiate the ELMo class

2. call the .load_pretrained_elmo() method with the desired .h5 file as its parameter

3. call the .get_elmo() function with any sentence you want to generate ELMo embeddings for. This will output a list of three embedded sentences(the index of the items in the list correspond to the layers of the architecture where the embeddings were pulled from)
Layer0 - output of the CNN
Layer1 - output of the first LSTM
Layer2 - output of the second LSTM

There is a function to visualize the quality of the embeddings of a single sentence through a heatmap of cosine similarities between all words.

For the operationalization of ELMo, the model needs to be trained extensively on the whole corpus, and a model needs to be saved to disk.
Once this is complete, a simple child class of ELMo needs to be implemennted that only loads a pretrained model and generates embeddings. This class could then be imported through the libraries and serve for all future or past NLP projects that are reliant on embeddings.

For further details on the functionality and implementation please refer to the docstrings.
