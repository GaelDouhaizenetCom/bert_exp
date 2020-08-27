from transformers import BertModel, BertTokenizer
import torch
from scipy.spatial.distance import cdist
import spacy
import numpy as np

# prepare distances selection,
# see also https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
distances = [
    'euclidean',
    # 'minkowski',
    'cityblock',
    # 'sqeuclidean',
    'cosine',
    # 'correlation',
    # 'hamming',
    # 'jaccard',
    # 'chebyshev',
    # 'canberra'
]

# pre-trained models loading
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = BertModel.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
nlp = spacy.load('en_core_web_sm')

# for example, obviously
test_sentence = 'The weather is quite nice today.'

# create the base embeddings
input_sentence = torch.tensor(tokenizer.encode(test_sentence)).unsqueeze(0)
out = model(input_sentence)
last_layer_mbdgs = out[0]
base_cls_mbdgs = last_layer_mbdgs[0][0]

# load text
doc = nlp("The weather was quite poor yesterday. The weather will be better tomorrow. Let's talk about something else.")

# iterate over the sentences
for sent in doc.sents:
    print(f'\n{test_sentence}\n{sent.text}')
    comp_sentence = torch.tensor(tokenizer.encode(sent.text)).unsqueeze(0)
    comp_last_layer_mbdgs = model(comp_sentence)[0]
    comp_cls_mbdgs = comp_last_layer_mbdgs[0][0]

    # compare embedding to the base cls token (used for classification)
    for d_type in distances:
        try:
            distance = cdist(np.array([base_cls_mbdgs.detach().numpy()]),
                             np.array([comp_cls_mbdgs.detach().numpy()]),
                             d_type)
            print('{} distance:\t {:0>.3f}'.format(d_type, np.squeeze(distance)))
        except Exception as e:
            print(str(type(e)), e.args[0])
