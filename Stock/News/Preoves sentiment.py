import stanza
nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment')
doc = nlp('I hate that they banned Mox Opal. i love my mom')
for i, sentence in enumerate(doc.sentences):
    print("%d -> %d" % (i, sentence.sentiment))