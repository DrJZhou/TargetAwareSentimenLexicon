import gzip
import codecs
import re
import numpy as np
from nltk import word_tokenize


num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
letter_regex = re.compile('^[a-z\']*$')

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield eval(l)

def preprocess_review(raw_review):

    review_text = ' '.join(word_tokenize(raw_review.lower()))
    #Replace smile emojis with SMILE
    review_text = re.sub(r'((?::|;|=)(?:-)?(?:\)|d|p))', " SMILE", review_text)
    review_text = re.sub(r' n\'t ', 'n\'t ', review_text)
    review_text = re.sub(r' \' t ', '\'t ', review_text)
    #Only keep letters, numbers, ', !, and SMILE
    words = []
    for w in review_text.split():
        if w in {'!', 'SMILE'}:
            words.append(w)
        elif bool(num_regex.match(w)):
            words.append(w)
        elif bool(letter_regex.match(w)):
            words.append(w)
    return ' '.join(words)


def extract_balenced_data(gen, out_path):
    out1 = codecs.open(out_path+'/data.txt', 'w', 'utf-8')
    out2 = codecs.open(out_path + '/train.tsv', 'w', 'utf-8')
    out3 = codecs.open(out_path + '/dev.tsv', 'w', 'utf-8')
    maxlen, count = 0, 0

    for review in gen:
        ## for amazon domain
        text = review["reviewText"]
        score = review['overall']

        ## for yelp data
        # text = review["text"]
        # score = review['stars']
        tokens = text.split()
        # if len(tokens) > maxlen_limit:
        #     continue
        count += 1
        if count %1000 == 0:
            print(count)
        if maxlen < len(tokens):
            maxlen = len(tokens)
        preprocessed = preprocess_review(text)
        out1.write(preprocessed + "\t" + str(int(score)-1) + '\n')
        if np.random.random() > 0.05:
            out2.write(preprocessed + "\t" + str(int(score)-1) + '\n')
        else:
            out3.write(preprocessed + "\t" + str(int(score)-1) + '\n')


if __name__ == "__main__":
    
    file_path = 'reviews_Electronics_5.json.gz'
    out_path = '.'

    # file_path = '../../domain_adaptation/amazon_dataset_large/yelp_review.json.gz'
    # out_path = 'yelp'

    gen = parse(file_path)
    extract_balenced_data(gen, out_path)


   

