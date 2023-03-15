from spacy_triple_extraction import spacy_NER
from utils import styled_print
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
def extract_entities(type, filenames):
    #extract all entities of `type` from csv file
    #create a histogram of each entity count
    #basically take out top n entities
    df = pd.DataFrame()
    df_list = []
    for filename in filenames:
        df_list.append(pd.read_csv(filename))
        styled_print(f"Loaded {filename}")
    df = pd.concat(df_list)
    entity_list = []
    for c, s in df.iterrows():
        entities = spacy_NER(s["paragraphs"], type)
        for entity in entities:
            entity_list.append(entity)
    return np.asarray(entity_list)


def entity_prefix_frequency(entity_list, n): #Returns dataframe of top n most frequent [0th] words in entity list
    entities_df = pd.DataFrame(entity_list)
    entities_np = np.array(entities_df[0].str.split().str.get(0))
    print(entities_np)
    unique, counts = np.unique(entities_np, return_counts=True) #Get counts for each unique entity
    unique_count = np.asarray((unique, counts)).T #Merge counts and entities
    # unique_count = unique_count[unique_count[:, 1].argsort()]
    top_n = pd.DataFrame(unique_count)
    top_n[1]=top_n[1].astype(int)
    top_n = top_n.sort_values(by=1)
    return top_n.tail(n)

def entity_frequency(entity_list, n): #Returns dataframe of top n most frequent 
    unique, counts = np.unique(entity_list, return_counts=True)
    unique_count = np.asarray((unique, counts)).T
    top_n = pd.DataFrame(unique_count)
    top_n[1]=top_n[1].astype(int)
    top_n = top_n.sort_values(by=1)
    return top_n.tail(n)
