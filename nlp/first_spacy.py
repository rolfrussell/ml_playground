import spacy
from spacy import displacy
nlp = spacy.load('en')
doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
          token.shape_, token.is_alpha, token.is_stop, token.is_punct)

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

displacy.serve(doc, style='dep')



nlp = spacy.load('en_core_web_md')  # make sure to use larger model!

tokens = nlp(u'dog cat banana')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))




doc = nlp(u"Peach emoji is where it has always been. Peach is the superior "
          u"emoji. It's outranking eggplant 🍑 ")
print(doc[0].text)          # Peach
print(doc[1].text)          # emoji
print(doc[-1].text)         # 🍑
print(doc[17:19].text)      # outranking eggplant

noun_chunks = list(doc.noun_chunks)
print(noun_chunks[0].text)  # Peach emoji

sentences = list(doc.sents)
assert len(sentences) == 3
print(sentences[1].text)    # 'Peach is the superior emoji.'
