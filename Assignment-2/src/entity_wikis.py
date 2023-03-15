from entity_analysis import extract_entities, entity_frequency
import fandom
from tqdm import tqdm
def get_wikis(entity_list):
    fandom.set_wiki("gameofthrones")
    wikis = []
    for ind, x in entity_list.iterrows():
        print(x[0])
        try:
            # wiki_page = fandom.page("Runespace")
            search = fandom.page(x[0])
            wikis.append(search.plain_text)
        except:
            print(f"COULDNT RETREIVE WIKI FOR {x[0]}")
    return wikis