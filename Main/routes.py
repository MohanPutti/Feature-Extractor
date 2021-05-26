from flask import render_template, request, url_for, redirect
from werkzeug import secure_filename
from Main import app
import nltk
import os
import io
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

from nltk.stem import WordNetLemmatizer
wnl1=WordNetLemmatizer()
from nltk.corpus import wordnet as wn
import numpy as np
contraction_map={
"ain't": "am not / are not / is not / has not / have not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"I'd": "I had / I would",
"I'd've": "I would have",
"I'll": "I shall / I will",
"I'll've": "I shall have / I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}


@app.route("/")
def index():
    return render_template('index.html')


def tokenize_text(text):
    tokens=nltk.word_tokenize(text)
    tokens=[tn.strip() for tn in tokens]
    return tokens



def expand_contractions(sent,contraction_map):
    pattern=re.compile('({})'.format('|'.join(contraction_map.keys())),flags=re.IGNORECASE|re.DOTALL)
    def expand_map(contraction):
        match=contraction.group(0)
        first_char=match[0]
        expansion=contraction_map.get(match) if contraction_map.get(match) else contraction_map.get(match.lower())
        expansion=first_char+expansion[1:]
        return expansion
    expand_sent=pattern.sub(expand_map,sent)
    return expand_sent


def pos_tag_text(text):
    def penn_to_wn_tags(pos_tag):
        if(pos_tag.startswith('ADJ')):
            return wn.ADJ
        elif(pos_tag.startswith('V')):
            return wn.VERB
        elif(pos_tag.startswith('N')):
            return wn.NOUN
        elif(pos_tag.startswith('ADV')):
            return wn.ADV
        else:
            return None
    tokens=tokenize_text(text)
    tokens_t=nltk.pos_tag(tokens,tagset='universal')
    #print(tokens_t)
    tokens_p=[(word.lower(), penn_to_wn_tags(pos_tag)) for word,pos_tag in tokens_t]
    #print(tokens_p)
    return tokens_p








def lemmatize_text(text):
    tokens_p=pos_tag_text(text)
    tokens_lm=[]
    for word, pos_tag in tokens_p:
        if pos_tag:
            tokens_lm.append(wnl1.lemmatize(word,pos_tag))
        else:
            tokens_lm.append(word)
    #print(tokens_lm)
    text_lm=" ".join(tokens_lm)
    return text_lm


def remove_special_characters(text):
    tokens=tokenize_text(text)
    pattern=re.compile('[{}]'.format(re.escape(string.punctuation)))
    tokens_f=filter(None,[pattern.sub('',tn) for tn in tokens])
    text_f=' '.join(tokens_f)
    return text_f

seq_sw1=nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    tokens=tokenize_text(text)
    tokens_f=[tn for tn in tokens if tn not in seq_sw1]
    text_f=' '.join(tokens_f)
    return text_f




def transform_corpus(myfile_sentences,tokenize=False):
    corpus_transformed=[]
    for sent in myfile_sentences:
        #print(sent)
        sent=expand_contractions(sent,contraction_map)
        sent=lemmatize_text(sent)
        sent=remove_special_characters(sent)
        
        sent=remove_stopwords(sent)
        corpus_transformed.append(sent)
        if( tokenize):
            sent=tokenize_text(sent)
            corpus_transformed.append(sent)
        #print(corpus_transformed)
    return corpus_transformed



def bow_extractor(corpus,ngram_range=(1,1)):
    vectorizer=CountVectorizer(min_df=1,ngram_range=ngram_range) #divides words into bags and count them in sentences
    features=vectorizer.fit_transform(corpus)
    return vectorizer,features

 ##TF-IDF Model
 # words with higher frequency are dominated but more releveant may be neglected in bag of model
def tfidf_transformer(bow_features):
    transformer=TfidfTransformer(norm='l2', smooth_idf=True,use_idf=True)
    features=transformer.fit_transform(bow_features)
    return (transformer,features)

@app.route("/extractFeatures",methods=['POST','GET'])
def dataCleaning():
    flask_file = request.files['file'] 
    myfile=flask_file.read()
    #print(myfile)
    myfile=myfile.decode('UTF-8')
    myfile_sentences=nltk.sent_tokenize(myfile)
    print(len(myfile_sentences))
    transformed_file=transform_corpus(myfile_sentences)
    #feature extraction using bag of model where the frequency of words dominate
    vectorizer_c,features_c=bow_extractor(transformed_file)
    features_cm=features_c.todense()
    feature_names=vectorizer_c.get_feature_names()
    df=pd.DataFrame(data=features_cm,columns=feature_names)
    print(df)
    #print(features_c)
    df.to_csv("D:/Projects/Major_Project/TextMining/features.csv",index=False)
    transformer_c,features_ct=tfidf_transformer(features_c)
    #features_ctm=np.round(features_ct.todense(),2)
    #df2=pd.DataFrame(data=features_ctm,columns=feature_names)
    #print(df2)
   

    return render_template('thankyou.html')

@app.route("/sendMail",methods=["POST",'GET'])
def sendMail():
    mail_content = '''Hello,
            This mail contains the features extracted from your file.
            Kindly find the attachment below.
    '''
    #The mail addresses and password
    sender_address = 'mohanputti9@gmail.com'
    sender_pass = '$$a_m_h_s_#999$$'
    receiver_address = request.form.get('email')
    print(receiver_address)
    #Setup the MIME
    message = MIMEMultipart()
    message['From'] = sender_address
    message['To'] = receiver_address
    message['Subject'] = 'Features attachment'
    #The subject line
    #The body and the attachments for the mail
    message.attach(MIMEText(mail_content, 'plain'))
    attach_file_name = 'features.csv'
    attach_file = open(attach_file_name, 'rb') # Open the file as binary mode
    payload = MIMEBase('application', 'octate-stream')
    payload.set_payload((attach_file).read())
    encoders.encode_base64(payload) #encode the attachment
    #add payload header with filename
    payload.add_header('Content-Decomposition', 'attachment', filename=attach_file_name)
    message.attach(payload)
    #Create SMTP session for sending the mail
    session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
    session.starttls() #enable security
    session.login(sender_address, sender_pass) #login with mail_id and password
    text = message.as_string()
    session.sendmail(sender_address, receiver_address, text)
    session.quit()
    print('Mail Sent')
    return render_template("mailsent.html")



