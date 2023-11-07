import streamlit as st
from sentence_transformers import SentenceTransformer, util
import re


def calculate_average_score(first_text: str, second_text: str, separators: list):
    if first_text == '' or second_text == '':
        return 0
    regex_separators = '[' + ''.join(separators) + ']'

    first_text_sentences = re.split(regex_separators, first_text)
    second_text_sentences = re.split(regex_separators, second_text)
    length_row = len(first_text_sentences)

    model = SentenceTransformer('sentence-transformers/LaBSE')

    first_embeddings = model.encode(first_text_sentences)
    second_embeddings = model.encode(second_text_sentences)

    cos_sim_arr = util.cos_sim(first_embeddings, second_embeddings)

    total_score = sum(map(lambda row: max(row), cos_sim_arr))
    average_score = total_score / length_row
    return average_score


