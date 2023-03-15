from entity_analysis import extract_entities, entity_frequency
import fandom
from tqdm import tqdm
def get_wikis(entity_list):
    wikis = []
    for x in tqdm(entity_list):
        try:
            wiki_page = fandom.page("Runespace")
            
        except:
            print(f"COULDNT RETREIVE WIKI FOR {x}")
        wikis.append(x)