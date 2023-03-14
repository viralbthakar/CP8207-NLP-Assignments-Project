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
    unique, counts = np.unique(entity_list[0], return_counts=True)
    unique_count = np.asarray((unique, counts)).T
    # unique_count = unique_count[unique_count[:, 1].argsort()]
    top_n = pd.DataFrame(unique_count)
    top_n[1]=top_n[1].astype(int)
    top_n = top_n.sort_values(by=1)
    return top_n.tail(n)

def entity_frequency(entity_list, n): #Returns dataframe of top n most frequent 
    unique, counts = np.unique(entity_list, return_counts=True)
    unique_count = np.asarray((unique, counts)).T
    # unique_count = unique_count[unique_count[:, 1].argsort()]
    top_n = pd.DataFrame(unique_count)
    top_n[1]=top_n[1].astype(int)
    top_n = top_n.sort_values(by=1)
    print(entity_prefix_frequency(entity_list, n))
    return top_n.tail(n)

print(entity_frequency(extract_entities("PERSON", ["Assignment-1/data/processed-data/clean-csvs/book-clean-paragraphs.csv", "Assignment-1/data/processed-data/clean-csvs/characters-clean-paragraphs.csv"]), 50))
# print(entity_frequency(extract_entities("GPE", ["Assignment-1/data/processed-data/clean-csvs/book-clean-paragraphs.csv", "Assignment-1/data/processed-data/clean-csvs/characters-clean-paragraphs.csv"]), 50))

