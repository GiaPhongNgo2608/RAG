import os
import json
import torch
import requests
import pandas as pd
from data import process_data
from embedding import Embedding
from index import Index
from bardapi import Bard, SESSION_HEADERS, BardCookies
import google.generativeai as genai
import google.generativeai as palm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def bard(prompt):
    session = requests.Session()
    token = "ewjZN5IpTSnK7R9oGXFKiuNHaKpbnWJyZaz664F5I1Nl-vok1Ekx6rrxT6Tvl2eonUNJCQ."
    session.cookies.set("__Secure-1PSID", "ewjZN5IpTSnK7R9oGXFKiuNHaKpbnWJyZaz664F5I1Nl-vok1Ekx6rrxT6Tvl2eonUNJCQ.")
    session.cookies.set("__Secure-1PSIDCC", "ABTWhQE7EFG34idEJ2gm2EGtWGOtki-dG5qlv4AlezR5Q2FJ9_ssPurM9pQkuvgfRI5XDn3DAlDq")
    session.cookies.set("__Secure-1PSIDTS", "sidts-CjIBPVxjSv92mLCVj8ehA1tSSRnlZcyaLRgU_v7l1PzvHwQQq6h2QrWgG_RQ8gByAosJgBAA")
    session.headers = SESSION_HEADERS

    bard = Bard(token=token, session=session)
    response = bard.get_answer(prompt)["content"]
    return response
def gemini(prompt):
    genai.configure(api_key="AIzaSyBoohaB85jrDff9FyXWvbqdcrqHaOEyJXE")
    model = genai.GenerativeModel('gemini-pro')
    chat = model.start_chat()
    response = chat.send_message(prompt)
    return response.text

def palm(prompt):
    palm.configure(api_key=os.environ["AIzaSyBoohaB85jrDff9FyXWvbqdcrqHaOEyJXE"])
    response = palm.generate_text(prompt="The opposite of hot is")
    print(response.result)

if __name__ == '__main__':

    file = '/home/rb025/RabilooAI/Transformer/RAG/2174_11_QD-DKVN~2020-05-13~1~Quy_trinh_xay.txt.json'
    # f = open(file)
    # data = json.load(f)
    # result = process_data(data)
    # text = " ".join(result)
    #model embed
    model_embed = Embedding(model_embedding_name='keepitreal/vietnamese-sbert',device=device,batch_size=32)
    query = 'Câu 1. Việc rà soát, cập nhật Chiến lược phát triển các đơn vị trực thuộc và các công ty con của Tập đoàn Dầu khí Việt Nam được thực hiện khi nào?'
    prompt = Index(file).retrieval(query,embed_model=model_embed())
    response = gemini(prompt)
    print(response)
    print("Done")
