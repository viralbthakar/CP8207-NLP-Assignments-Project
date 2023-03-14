import spacy
from spacy import displacy
nlp_model = spacy.load("en_core_web_sm") 
"""spacy.load will "load" a pretrained
 model which can perform tokenization, 
 POS tagging, depency parsing, NER, text categorizer(sentiment, spam),
 rule based matcher(like regex), entity linker, more. In our case the 
 en_core_web_sm(english core pipieline features, trained on web, small) has 
 tok2vec, tagger, parser, attribute_ruler, lemmatizer, ner"""


our_text = "Jessica M Roberts lives in Canada"
doc = nlp_model(our_text)

def spacy_NER(input_str, label):
    annotated_str = {}
    for entity in nlp_model(input_str).ents:
        if entity.label_ == label:
            annotated_str[entity.text] = entity.label_
    return annotated_str
def sort_entities(entity_dict):
    categories = ['ORG', 'CARDINAL', 'DATE', 'GPE', 'PERSON', 'MONEY', 'PRODUCT', 'TIME', 'PERCENT', 'WORK_OF_ART', 'QUANTITY', 'NORP', 'LOC', 'EVENT', 'ORDINAL', 'FAC', 'LAW', 'LANGUAGE']
    categories = {x:0 for x in categories}
    for x in entity_dict:
        categories[entity_dict[x]]+=1
    return categories
def subject_object_predicate(doc, entities):
    subject = {}
    object = {}
    predicate = {}
    for token_n, token in enumerate(doc):
        #This will lead to compounds from one entity to be in the other! But we can clean those out later
        if "compound" in token.dep_:
            subject[token.text] = token.idx
        if "subj" in token.dep_:
            subject[token.text] = token.idx
        
        #This will lead to compounds from one entity to be in the other! But we can clean those out later
        if "compound" in token.dep_:
            object[token.text] = token.idx
        if "obj" in token.dep_:
            object[token.text] = token.idx


        if "verb" in token.pos_.lower() or "root" in token.dep_.lower():
            predicate[token.text] = token.idx
            if "prep" in doc[token_n+1].dep_:
                predicate[doc[token_n+1].text] = doc[token_n+1].idx
    
    #clean object and subject 
    #For each entity
    for entity in entities:
        #Split entity into a list
        entity_list = entity.split(" ")
        #For the object, find which words in it match the current entity
        object_match = [x for x in list(object.keys()) if x in entity_list]
        #If it's not a perfect match, then the object has compounds in it that don't exist in the actual entity it's associated with! So we should remove those matching words
        if entity_list != object_match:
            for word in object_match:
                try:
                    object.pop(word)
                except:
                    None
        #For the subject, find which words in it match the current entity
        subject_match = [x for x in list(subject.keys()) if x in entity_list]
        #If it's not a perfect match, then the subject has compounds in it that don't exist in the actual entity it's associated with! So we should remove those matching words
        if entity_list != subject_match:
            for word in subject_match:
                try:
                    object.pop(word)
                except:
                    None
    return {"subject":subject, "object":object, "predicate":predicate}

def spacyRule_triple_extraction_pipeline(sentence):
    doc = nlp_model(sentence)
    entities = spacy_NER(sentence)
    print(entities)
    triple = subject_object_predicate(doc, entities)
    print(f"({' '.join(list(triple['subject']))})---({' '.join(list(triple['predicate']))})--->({' '.join(list(triple['object']))})")
    return subject_object_predicate(doc, entities)