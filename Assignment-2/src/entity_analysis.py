from spacy_triple_extraction import spacy_NER
import sentiment_analysis
from utils import styled_print
import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from collections import defaultdict


import spacy
import en_core_web_sm

nlp = spacy.load('en_core_web_sm')

def extract_entities(type, filenames): #returns numpy array of entitys of type TYPE, and 
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
    for ind, s in df.iterrows():
        entities = spacy_NER(s["paragraphs"], type)
        for entity in entities:
            entity_list.append(entity)
    return np.asarray(entity_list)

def extract_all_entities(type, filenames): #returns numpy array of entitys of type TYPE, and 
    #extract all entities of `type` from csv file
    #create a histogram of each entity count
    #basically take out top n entities
    df = pd.DataFrame()
    df_list = []
    for filename in filenames:
        df_list.append(pd.read_csv(filename))
        styled_print(f"Loaded {filename}")
    df = pd.concat(df_list)
    entity_list = defaultdict(list)

    for ind, s in df.iterrows():
        entities = spacy_NER(s["paragraphs"], type)
        for entity in entities:
            entity_list[entity] = entities[entity]
    return entity_list




def find_non_person_entity_sentences(entities, filenames):
    non_person_entity_annotated_sentences = {}
    df = pd.DataFrame()
    df_list = []
    for filename in filenames:
        df_list.append(pd.read_csv(filename))
        styled_print(f"Loaded {filename}")
    df = pd.concat(df_list)
    for ind, sentence in tqdm(df.iterrows()):
        for entity in entities:
            if entity in sentence[1]:
                non_person_entity_annotated_sentences[sentence[1]] = (entity)
    return non_person_entity_annotated_sentences





def pairwise_extract_sentiment(pair_annotated_sentences): #Returns a dictionary like {(entity, entity), [sentiment1, sentiment2, sentimentn]}
    character_pair_sentiment_attributes = defaultdict(list)
    for sentence in tqdm(pair_annotated_sentences):
        nlp_doc = nlp(sentence)
        subject = [tok.text for tok in nlp_doc if (tok.dep_ == "nsubj")]
        object = [tok.text for tok in nlp_doc if (tok.dep_ == "dobj")]
        for potential_entity in subject:
            if str(potential_entity)==pair_annotated_sentences[sentence][0]:
                if pair_annotated_sentences[sentence][1] in object:
                    character_pair_sentiment_attributes[pair_annotated_sentences[sentence]].append(sentiment_analysis.sentiment_analysis(sentence))
           
            elif str(potential_entity)==pair_annotated_sentences[sentence][1]:
                if pair_annotated_sentences[sentence][0] in object:
                    character_pair_sentiment_attributes[pair_annotated_sentences[sentence]].append(sentiment_analysis.sentiment_analysis(sentence))
    return character_pair_sentiment_attributes




def non_person_entity_extract_sentiment(entity_annotated_sentences): 
    entity_sentiment_attributes = defaultdict(list)
    for sentence in tqdm(entity_annotated_sentences):
        entity_sentiment_attributes[entity_annotated_sentences[sentence][0]].append(sentiment_analysis.sentiment_analysis(sentence))
    return entity_sentiment_attributes




def person_extract_sentiment(pair_annotated_sentences):
    character_sentiment_attributes = defaultdict(list)
    for sentence in tqdm(pair_annotated_sentences):
        nlp_doc = nlp(sentence)
        subject = [tok.text for tok in nlp_doc if (tok.dep_ == "nsubj")]
        for potential_entity in subject:
            if str(potential_entity)==pair_annotated_sentences[sentence][0]:
                character_sentiment_attributes[potential_entity].append(sentiment_analysis.sentiment_analysis(sentence))
            elif str(potential_entity)==pair_annotated_sentences[sentence][1]:
                character_sentiment_attributes[potential_entity].append(sentiment_analysis.sentiment_analysis(sentence))
    return character_sentiment_attributes




def average_sentiments(character_sentiment_attributes):
    mean_character_sentiment_attributes = {}
    for character in character_sentiment_attributes:
        if len(character_sentiment_attributes[character]) > 0:
            mean_character_sentiment_attributes[character] = sum(character_sentiment_attributes[character])/len(character_sentiment_attributes[character]) #Find the mean of their sentiments
    return mean_character_sentiment_attributes





def find_entity_pair_sentences(entity_pairs, filenames):
    pair_annotated_sentences = {}
    df = pd.DataFrame()
    df_list = []
    for filename in filenames:
        df_list.append(pd.read_csv(filename))
        styled_print(f"Loaded {filename}")
    df = pd.concat(df_list)
    for ind, sentence in tqdm(df.iterrows()):
        for pair in entity_pairs:
            if pair[1] in sentence[1] and pair[0] in sentence[1]:
                pair_annotated_sentences[sentence[1]] = pair
    return pair_annotated_sentences





def entity_prefix_frequency(entity_list, n): #Returns dataframe of top n most frequent [0th] words in entity list
    entities_df = pd.DataFrame(entity_list)
    entities_np = np.array(entities_df[entities_df[0].astype(str).str.len() > 10][0].str.split().str.get(0))
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


def entity_occurences(entities, filenames):
    entity_occurences = defaultdict(list)
    df = pd.DataFrame()
    df_list = []
    for filename in filenames:
        df_list.append(pd.read_csv(filename))
        styled_print(f"Loaded {filename}")
    df = pd.concat(df_list)
    print(df)
    full_text = df["paragraphs"].str.cat(sep=' ')
    n=0
    for word in tqdm(full_text.split()):
        for entity in entities:
            if word==entity:
                entity_occurences[entity].append(n)
        n+=1
    return entity_occurences
