#!/usr/bin/env python
# coding: utf-8

# In[8]:


import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
    print(e)


# In[1]:


# 챗봇 패키지
import streamlit as st
from streamlit_chat import message
import json

# 공통
import torch
import numpy as np
import pandas as pd
import kss
import re

from tqdm import tqdm, tqdm_notebook
import tqdm

# GPT2
#import tensorflow as tf
from transformers import TFGPT2LMHeadModel, AutoTokenizer
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

# KoBERT
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model


# In[11]:


# 학습 데이터 전처리 클래스
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))


# In[12]:


# KoBERT 모델 클래스(분류기)
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=3,   ##클래스 수 조정##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


# In[13]:


# vocabulary 불러오기
@st.cache(allow_output_mutation=True)
def cached_koBERT_vocab():
    _, vocab = get_pytorch_kobert_model()
    return vocab


# In[14]:


# 모델 캐시
@st.cache(allow_output_mutation=True)
def cached_koBERT():
    
    model = torch.load('./model/KoBERT_cls_model.pt')
    model.eval()
    
    return model


# In[15]:


#GPU 사용
device = torch.device("cuda:0")


# In[16]:


# params 설정
max_len = 64
batch_size = 64


# In[17]:


# 토크나이저 로드
@st.cache(allow_output_mutation=True)
def cached_koBERT_tok():
    tokenizer_kobert = get_tokenizer()
    tok_kobert = nlp.data.BERTSPTokenizer(tokenizer_kobert, vocab, lower=False)
    
    return tok_kobert


# In[18]:


vocab = cached_koBERT_vocab()
model_kobert = cached_koBERT()
tok_kobert = cached_koBERT_tok()


# In[ ]:


# 감정 예측 함수
def predict_emotion(predict_sentence):
    
    result_emo = None
    
    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok_kobert, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)
    
    #model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length= valid_length
        label = label.long().to(device)

        out = model_kobert(token_ids, valid_length, segment_ids)


        test_eval=[]
        for i in out:
            logits=i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append("분노가")
                result_emo = '분노'
            elif np.argmax(logits) == 1:
                test_eval.append("행복이")
                result_emo = '행복'
            elif np.argmax(logits) == 2:
                test_eval.append("슬픔이")
                result_emo = '슬픔'
                
        print(">> 입력하신 내용에서 " + test_eval[0] + " 느껴집니다.")
    
    return result_emo


# In[ ]:


# 토크나이저 로드
@st.cache(allow_output_mutation=True)
def cache_gpt2_tok():
    tok = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>', pad_token='<pad>')
    return tok


# In[ ]:


# gpt2_title_model 로드
@st.cache(allow_output_mutation=True)
def cache_gpt2_title():
    model = TFGPT2LMHeadModel.from_pretrained('./model/Gen_title_GPT2_model.h5')
    return model


# In[ ]:


gpt2_tokenizer = cache_gpt2_tok()
gpt2_title_model = cache_gpt2_title()


# In[ ]:


# 제목 생성
def gen_title(user_emotion):
    sent = '<usr>' + user_emotion + '<sys>'
    input_ids = [gpt2_tokenizer.bos_token_id] + gpt2_tokenizer.encode(sent)
    input_ids = tf.convert_to_tensor([input_ids])
    output = gpt2_title_model.generate(input_ids, max_length=50, do_sample=True, temperature=0.85, top_p=0.80)
    sentence = gpt2_tokenizer.decode(output[0].numpy().tolist())
    gened_title = sentence.split('<sys> ')[1].replace('</s>', '')
    return gened_title


# In[ ]:


# gpt2_sent_model 로드
@st.cache(allow_output_mutation=True)
def cache_gpt2_sent():
    model = TFGPT2LMHeadModel.from_pretrained('./model/Gen_sent_GPT2_model.h5')
    return model


# In[ ]:


gpt2_sent_model = cache_gpt2_sent()


# In[ ]:


# 문장생성
def gen_sent(user_emotion):
    sent = '<usr>' + user_emotion + '<sys>'
    input_ids = [gpt2_tokenizer.bos_token_id] + gpt2_tokenizer.encode(sent)
    input_ids = tf.convert_to_tensor([input_ids])
    output = gpt2_sent_model.generate(input_ids, max_length=50, do_sample=True, temperature=0.85, top_p=0.80)
    sentence = gpt2_tokenizer.decode(output[0].numpy().tolist())
    gened_sent = sentence.split('<sys> ')[1].replace('</s>', '')
    return gened_sent


# In[ ]:


@st.cache(allow_output_mutation=True)
def cache_gpt2_tok2():
    tok = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                                    pad_token='<pad>', mask_token='<mask>')
    return tok


# In[ ]:


@st.cache(allow_output_mutation=True)
def cache_gpt2_novel():
    model = GPT2LMHeadModel.from_pretrained('./model/Gen_novel_GPT2_model')
    model.eval()
    
    return model


# In[ ]:


gpt2_novel_tokenizer = cache_gpt2_tok2()


# In[ ]:


gpt2_novel_model = cache_gpt2_novel()


# In[ ]:


def gen_novel(usr_txt):
    input_ids = gpt2_novel_tokenizer.encode(usr_txt)
    gen_ids = gpt2_novel_model.generate(torch.tensor([input_ids]),
                               max_length=512, # generate 할 개수
                               repetition_penalty=2.0, # 단어 반복시 패널티를 주어서 새로운 단어를 생성
                               temperature=0.85,
                               top_p=0.80,
                               pad_token_id=gpt2_novel_tokenizer.pad_token_id,
                               eos_token_id=gpt2_novel_tokenizer.eos_token_id,
                               bos_token_id=gpt2_novel_tokenizer.bos_token_id,
                               use_cache=True)
    generated = gpt2_novel_tokenizer.decode(gen_ids[0,:].tolist())
    generated = kss.split_sentences(generated)
    
    tmp_list = []
    for i in generated:
        if i[-1] == '.':
            tmp_list.append(i)
    generated = "\n".join(tmp_list)
    
    return generated


# In[ ]:


st.header('AI진용진')
st.markdown("[당신을 위한 이 세상에 없는 소설을 알려드림.]")


# In[ ]:
if 'emotion' not in st.session_state:
    st.session_state['emotion'] = []

if 'title' not in st.session_state:
    st.session_state['title'] = []
    
if 'generated' not in st.session_state:
    st.session_state['generated'] = []


# In[ ]:


if 'past' not in st.session_state:
    st.session_state['past'] = []


# In[ ]:


with st.form('form', clear_on_submit=True):
    user_input = st.text_input('당신: ', '')
    submitted = st.form_submit_button('전송')


# In[ ]:


if submitted and user_input:
    
    # 유저 채팅의 감정 파악
    emotion = predict_emotion(user_input)
    message(f'당신에게서 {emotion}의 감정이 느껴집니다.', key='emobot')
    message(f'지금부터 {emotion}의 소설을 작성하겠습니다....', key='emo2bot')
    
    # 감정에 맞는 제목 설정(테스트 : 임의 설정)
    title = gen_title(emotion)
    #message(f"<{title}>", key='tbot')
    # 감정에 맞는 첫 문장 설정(테스트 : 임의 설정)
    first_sentence = gen_sent(emotion)
    first_sentence = re.sub('[---p.*0-9]', '', first_sentence)
    
    # GPT3로 소설을 작성했다고 가정 -> 문장 띄워쓰기 해결해야함.
    generated = gen_novel(first_sentence)
    generated = generated.replace('<unk>', '')


    st.session_state.emotion.append(emotion)
    st.session_state.title.append(title)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(generated)


# In[ ]:


for i in range(len(st.session_state['past'])):
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    if len(st.session_state['generated']) > i:
        message(f"{st.session_state['emotion'][i]}의 영감을 받아 적은 작품입니다.", key=str(i) + '_gbot')
        message(f"<{st.session_state['title'][i]}>", key=str(i) + '_gggbot')
        message(st.session_state['generated'][i], key=str(i) + '_bot')

