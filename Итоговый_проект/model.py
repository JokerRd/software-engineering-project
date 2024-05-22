from sentence_transformers import SentenceTransformer, util
import re


def calculate_average_score(first_text: str, second_text: str,
                            separators: list):
    cos_sim_arr, first_text_sentences, second_text_sentences = compare_text(
        first_text, second_text, separators)
    length_row = len(first_text_sentences)
    total_score = sum(map(lambda row: max(row), cos_sim_arr))
    average_score = total_score / length_row
    return average_score


def get_stat_by_sentences(first_text: str, second_text: str, separators: list):
    cos_sim_arr, first_text_sentences, second_text_sentences = compare_text(
        first_text, second_text, separators)
    result = []
    for i in range(len(cos_sim_arr)):
        for j in range(len(cos_sim_arr[i])):
            result.append({
                "first_sentence": first_text_sentences[i],
                "second_sentence": second_text_sentences[j],
                "similarity": float(cos_sim_arr[i][j])
            })

    return result


def compare_text(first_text: str, second_text: str, separators: list):
    if first_text == '' or second_text == '':
        return 0
    regex_separators = '[' + ''.join(separators) + ']'

    first_text_sentences_raw = re.split(regex_separators, first_text)
    second_text_sentences_raw = re.split(regex_separators, second_text)
    first_text_sentences = trim_spaces(
        filter_empty_sentences(first_text_sentences_raw))
    second_text_sentences = trim_spaces(
        filter_empty_sentences(second_text_sentences_raw))

    model = SentenceTransformer('sentence-transformers/LaBSE')

    first_embeddings = model.encode(first_text_sentences)
    second_embeddings = model.encode(second_text_sentences)

    cos_sim_arr = util.cos_sim(first_embeddings, second_embeddings)
    return cos_sim_arr, first_text_sentences, second_text_sentences


def filter_empty_sentences(sentences: list[str]):
    return list(filter(lambda item: item != '', sentences))


def trim_spaces(sentences: list[str]):
    return list(map(lambda item: item.strip(), sentences))
