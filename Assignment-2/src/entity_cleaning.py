from itertools import combinations
from entity_analysis import extract_entities
import Levenshtein

def entity_pairs(entity_list):
    pairs = set(list(combinations(entity_list, 2))) 
    return pairs

def longest_str(pair):
     if pair[0] > pair[1]:
          return pair[0]
     else:
          return pair[1]

def clean_pairs(entity_pairs, entity_list): #Removes potential duplicates from pairs
    dists = {}
    for pair in entity_pairs:
            dists[pair] = (Levenshtein.ratio(pair[1], pair[0]))
            if dists[pair] > 0.7:
                entity_list.remove(longest_str(pair))
    character_attributes = {}