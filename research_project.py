import streamlit as st
st.title('Research Paper Title generation')
abstract=st.text_area("Input abstract here to generate title")
import os
import torch
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation

nlp = spacy.load('en_core_web_sm')
if st.button("Generate"):
    from transformers import GPT2LMHeadModel,GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    def generate_title(prompt_text, num_titles, max_length=50):
        if not prompt_text:
            raise ValueError("prompt_text must not be empty")
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    
        # Generate titles with constrained length
        output_ids = model.generate(
            input_ids,
            do_sample=True,
            max_length=max_length,
            #min_length=25,
            #max_new_tokens=512,
            num_return_sequences=num_titles,
            top_k=10,  # Reduce to focus on specific outputs
            top_p=0.7,  # Adjust top-p to reduce randomness
            temperature=0.5,  # Lower temperature for more coherent output
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=True,  # Stop early to avoid excessive output
        )

        # Decode and trim the generated titles
        titles = [
            tokenizer.decode(ids, skip_special_tokens=True).split('\n')[0]  # Keep only the first line
            for ids in output_ids
        ]
        return titles
    doc=nlp(abstract)
    dct = {}
    for i in doc:
        dct[i.text.lower()] = 0
    for i in doc:
        dct[i.text.lower()] += 1
    sw = STOP_WORDS
    sw.add("specifically")
    sw.add("magenta")
    dct = {i:v for i,v in dct.items() if i not in sw and i not in punctuation}
    max_freq = sorted(dct.items(), key=lambda x: x[1], reverse=True)[0][1]

    for i in dct.keys():
        dct[i] /= max_freq
    sent_token = [i for i in doc.sents]
    score = {}
    for sent in sent_token:
        score[sent] = 0
    for sent in sent_token:
        for word in sent:
            if word.text in dct.keys():
                score[sent] += dct[word.text]
    from heapq import nlargest
    len(score)*.3

    summary = nlargest(n=4, iterable=score, key=score.get)

    l=[]
    for i in summary:
        l.append(str(i))
    titles=[]
    for i in l :
        titles.append(generate_title(i, num_titles=5))
    dct={}
    doc=[]
    for i in titles:
        doc.append(nlp(i[0]))

    for i in doc:
        for j in i:
            dct[j.text.lower()] = 0
    for i in doc:
        for j in i:
            dct[j.text.lower()] += 1
    sw = STOP_WORDS
    dct = {i:v for i,v in dct.items() if i not in sw and i not in punctuation}
    max_freq = sorted(dct.items(), key=lambda x: x[1], reverse=True)[0][1]
    
    for i in dct.keys():
        dct[i] /= max_freq
    title_words=[]
    max_val=0.0
    for i in dct:
        if max_val<dct[i] and dct[i]!=1.0:
            max_val=dct[i]
    
    for i in dct :
        if dct[i]>=(max_val/2) :
            title_words.append(i)
    title_words=title_words[:]
    text=" ".join(title_words[:int((len(title_words)/2))])
    st.text_area(label ="Use these words to write your title",value=text, height =50)