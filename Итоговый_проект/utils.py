import copy


def transform_similarity_to_percent(stat):
    similarity_as_number = stat['similarity']
    similarity_as_percent = convert_to_percent(similarity_as_number)
    new_stat = copy.deepcopy(stat)
    new_stat['similarity'] = similarity_as_percent
    return new_stat


def convert_to_percent(similarity: float):
    rounded_percent_similarity = round(similarity * 100, 2)
    return f"{rounded_percent_similarity}%"
