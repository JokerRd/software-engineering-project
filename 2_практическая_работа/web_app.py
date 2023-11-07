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


def compare_text(first_text: str, second_text: str, options: list):
    separators = list(map(lambda option: option_to_separator(option), options))
    average_score = calculate_average_score(first_text, second_text, separators)
    average_score_percent = average_score * 100

    st.text("Тексты похожи на {:.2f} %".format(average_score_percent))


def option_to_separator(option: str):
    if option == 'Точка':
        return '.'
    if option == 'Вопросительный знак':
        return '?'
    if option == 'Восклицательный знак':
        return '!'
    if option == 'Точка с запятой':
        return ';'
    if option == 'Запятая':
        return ','
    if option == 'Тире':
        return '-'
    return None


st.title('Антиплагиат')

st.write('Это приложение позволяет сравнить два текста между собой по предложениям и определяет '
        'похожесть текстов друга на друга.')
st.write('Для работы выберите какими разделителями разделены предложения, введите'
        'последовательно два текста, которые хотите сравнить в соответствующие окна и нажмите кнопку сравнить. '
        'Будет отображаться процент похожести текстов друг на друга.')

options = st.multiselect(
    'Выберите разделитель предложений',
    ['Точка', 'Вопросительный знак', 'Восклицательный знак', 'Точка с запятой', 'Запятая', 'Тире'],
    ['Точка', 'Вопросительный знак', 'Восклицательный знак'])

col1, col2 = st.columns(2)

with col1:
    first_text_area = st.text_area('Введите первый текст для сравнения')

with col2:
    second_text_area = st.text_area('Введите второй текст для сравнения')

button = st.button("Сравнить", on_click=compare_text(first_text_area, second_text_area, options))

