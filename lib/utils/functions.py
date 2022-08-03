import pandas as pd
import numpy as np
import spacy
from spacy import displacy
import networkx as nx
import matplotlib.pyplot as plt
import os
import re
import scipy
from pyvis.network import Network
import community as community_louvain


def import_books(path):

    """
    imports every a song of ice and fire book in text (.txt) format from the given path

    params:
    path -- the path where the books are, if mac the folder where the book are (asuming that the folder is inside it' inside this project folder)

    returns:
    a list of the books

    """

    all_books = [b for b in os.scandir(path) if '.txt' in b.name]

    return all_books


def NER(book_name):

    """
    NER stands for name entity recognition

    process text from a text file using Spacy to recognize the entities within it

    params:
    file_name -- name of a txt file as string or position of the book in a list containing every book

    returns:
    processed doc file using Spacy English language model

    """

    NLP = spacy.load('en_core_web_sm') #Spacy English language model
    NLP.max_length = 2300000

    book_text = open(book_name, encoding='utf-8').read()
    book_doc = NLP(book_text)
    
    return book_doc


def get_ne_list_per_sentence(book_doc):

    """
    create a list of entites per sentence of a Spacy document and store in a dataframe.

    params:
    book_doc -- a spacy processed document

    returns:
    a dataframe containing the sentences and corresponding list of recognised named entities in the sentences

    """

    sent_entity_df = []

    #loop through sentences, store named entities for each sentence

    for sent in book_doc.sents:
        entity_list = [ent.text for ent in sent.ents]
        sent_entity_df.append({'sentence': sent, 'entities': entity_list})

    sent_entity_df = pd.DataFrame(sent_entity_df)

    return sent_entity_df


def import_character_list(file_name):

    """
    imports a csv with every named character in the a song of ice of fire book series

    params:
    file_name -- the name of the csv where the characters dataframe

    returns:
    characters dataframe

    """

    characters_df = pd.read_csv(file_name)
    
    #remove the parentesis from character names in the df 
    characters_df['name'] = characters_df['name'].apply(lambda x: re.sub('[\(.*?)]', '', x))
    
    #creating a column for the first name of the characters
    characters_df['first_name'] = characters_df['name'].apply(lambda x: x.split(' ', 1)[0])

    return characters_df


def filter_entity(ent_list, characters_df):

    """

    filter non-character entities.

    params:
    ent_list -- list of entities to be filtered
    character_df -- a dataframe contain characters' names and characters' first names

    returns:
    a list of entities that are characters (matching by names or first names)

    """

    return [ent for ent in ent_list
           if ent in list(characters_df.name)
           or ent in list(characters_df.first_name)]


"""
to apply the function use:
sent_entity_df['character_entities'] = sent_entity_df['entities'].apply(lambda x: filter_entity(x, characters_df))

and to filter sentences without entities use:
sent_entity_df_filtered = sent_entity_df[sent_entity_df['character_entities'].map(len) > 0]
"""


def create_relationships(df, window_size):

    """
    creates a dataframe of relationships based on the df dataframe (containing lists of chracters per sentence) and the window size of n sentences

    params:
    df -- a dataframe containing a column called character_entities with the list of chracters for each sentence of a document
    window_size -- size of the windows (number of sentences) for creating relationships between two adjacent characters in the text

    returns:
    a relationship dataframe containing 3 columns: source, target, value

    """

    relationships = []

    for i in range(df.index[-1]):
        end_i = min(i + 5, df.index[-1]) #this will avoid an out of range error in case i+5 exceeds the last row of the df
        char_list = sum((df.loc[i:end_i].character_entities),[]) #storing the entities that appear in the window size by calling a sum function and merging the list with an empty one

        #remove duplicated characters in the same sentence
        #check if it's the first time the character appears in the sentence or if the character it's not same as the one that appeared before
        unique_char = [char_list[i] for i in range(len(char_list)) if (i==0) or char_list[i] != char_list[i-1]]

        #checking if there is more than one character
        if len(unique_char) > 1:
            #look at each character that appears in the window and create a df with the relationship between then
            for idx, a in enumerate(unique_char[:-1]): #iterating until the second last character to not exceed the index of the list
                b = unique_char[idx + 1]
                relationships.append({'source': a, 'target': b})

    #aggregating all the duplicated relationships, this will include reverser relationships; Eddard Stark\tBran and Bran\tEddard Stark for example

    #sort the df to identify the duplicates
    relationships = pd.DataFrame(np.sort(relationships_df.values, axis=1), columns = relationships_df.columns)
    
    #adding a weight to the relation ship
    relationships['value'] = 1
    #summing up all the weights of the relationships
    relationships = relationships.groupby(['source', 'target'], sort=False, as_index=False).sum()

    return relationships


def graph_of_degreess(degree_dict):

    """
    creates a graph degreess of centrality of the characters in the book

    params:
    degree_dict -- a dictionary of the degreess of centrality of the characters (can be used with closeness and betweenness)

    returns:
    a bar plot of the 10 most central characters

    """

    degree_df = pd.DataFrame.from_dict(degree_dict, orient='index', columns=['centrality'])

    degree_df.sort_values('centrality', ascending=False)[0:9].plot(kind='bar')

    plt.show()


def relationships_graph(relationships):

    """
    creates a graph of the relationship using network x and then pyvis to visualize the communities in the graph

    params:
    relationships -- a df with the columns source: source character of the interaction, target: target character of the interaction, value: number of interactions

    returns:
    graph of the relationship of the characters in the books

    """

    #creating the visualization of the relationship_df
    #this is a simplistic model so the only edge attribute to take into account will be the value column of the data frame
    G = nx.from_pandas_edgelist(relationships, source='source', target='target', edge_attr='value', create_using = nx.Graph())

    plt.figure(figsize=(20,20))
    pos = nx.kamada_kawai_layout(G) #I like how this one looks... spoilers to many characters so it looks horrible
    nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos=pos)
    plt.show()

    #degree of centralityn"
    degree_dict = nx.degree_centrality(G)

    #closeness centrality"
    closeness_dict = nx.closeness_centrality(G)

    #betweeness centrality"
    betweenness_dict = nx.betweenness_centrality(G)

    #community detection"
    communities = community_louvain.best_partition(G)

    #using pyvis to make the visualization bearable to the eyes
    net = Network(notebook=True, width='1000px', height='700px', bgcolor='#222222', font_color='white')
    net.repulsion()

    #sizing the nodes by their degrees (how many conections they have)
    node_degree = dict(G.degree)
    nx.set_node_attributes(G, node_degree, 'size')
    nx.set_node_attributes(G, communities, 'group')

    #pyvis interfaces with network X"
    net.from_nx(G)
    net.show('G.html')

    graph_of_degreess(degree_dict)
    graph_of_degreess(closeness_dict)
    graph_of_degreess(betweenness_dict)