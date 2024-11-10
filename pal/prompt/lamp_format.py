import json
import random




# citation choice, "classification"
lamp1_white_prompt_words = [
    # original
    "Write a summary, in English, of the research interests and topics of a researcher who has published the following papers. Only generate the summary, no other text.", 

    "Your task is to generate a summary, in English, of the research interests and topics of a researcher who has published the following papers listed in the profile. The summary will be used to answer only the last question. You should not provide the direct answer to the question. Instead, focus on identifying the most relevant key points in the profile for telling apart the two references.",
    # summary
    "Your task is to generate a summary, in English, of the research interests and topics of a researcher who has published the following papers listed in the profile. The summary will be used to answer only the last question. You should not provide the direct answer to the question. Instead, focus on identifying the most relevant key points in the profile for telling apart the two references. Use the following format: The researcher's research interests and topics are: 1. '[research interest and topic 1]' 2. '[research interest and topic 2]' ... N. '[research interest and topic N]'.",
    # rationale
    "Your task is to generate a set of rationales for answering the last question, in English, that outline the steps or information needed to arrive at the answer. You are provided with a summary of the researcher's research interests and topics. You should not provide the direct answer to the question. Instead, focus on synthesizing and comparing the relevant information from the summary with the two references in the question. Use the following format: The rationale for answering the question is: 1. '[rationale for reference 1]' 2. '[rationale for reference 2]'.",
    # summary and rationale
    "Your task is: first, generate a summary, in English, of the research interests and topics of a researcher who has published the following papers listed in the profile. The summary will be used to answer only the last question. You should not provide the direct answer to the question. Instead, focus on identifying the most relevant key points in the profile for telling apart the two references. Then, generate a set of rationales for answering the last question, in English, that outline the steps or information needed to arrive at the answer based on the summary. You should not provide the direct answer to the question. Instead, focus on synthesizing and comparing the relevant information from the summary with the two references in the question. Use the following format: The rationale for answering the question is: 1. '[rationale for reference 1]' 2. '[rationale for reference 2]'. Focus on the last question provided in the prompt. Output the summary and rationale in a single response."
    ]

# to be updated
lamp2N_white_prompt_words = [
    # summary
    "Look at the following past articles this journalist has written and determine the most popular category they write in. Answer in the following format: most popular category: <category top1>, <category top2>, ..., <category topn>."
]

# to be updated
lamp2M_white_prompt_words = [
    # summary
    "Look at the following past movies this user has watched and determine the mostpopular tag they labeled. Answer in the following form: most popular tag: <tag top1>, <tag top2>, ..., <tag topn>."
]

# product review rating, "classification"
lamp3_white_prompt_words = [
    "Based on this user's past reviews, what are the most common scores they give for positive and negative reviews? Answer in the following form: most common positive score: <most common positive score>, most common negative score: <most common negative score>",
    # short
    "Your task is to generate a summary based on the user's past reviews, in English, of the user's most common scores given for positive and negative reviews. The summary will be used to answer only the last question. You should not provide the direct answer to the question. Answer in the following format: The user's most common scores given for positive and negative reviews are: 1. '[most common positive score: <most common positive score>]' 2. '[most common negative score: <most common negative score>]'.",
    # score and attitude
    "Your task is to generate a summary based on the user's past reviews, in English, of the user's common scores given for positive and negative reviews and the corresponding attitudes of the user. The summary will be used to answer only the last question. You should not provide the direct answer to the question. Instead, focus on identifying the key points for mapping the attitude of the user to the scores given for positive and negative reviews. Use the following format: The user's common scores given for positive and negative reviews and the corresponding attitudes are: 1. '[Top1 most common positive score: <Top1 most common positive score>; Attitude: <Attitude>]' 2. '[Top2 most common positive score: <Top2 most common positive score>; Attitude: <Attitude>]' ... 1. '[Top1 most common negative score: <Top1 most common negative score>; Attitude: <Attitude>]' 2. '[Top2 most common negative score: <Top2 most common negative score>; Attitude: <Attitude>]' ... Focus only on the last question provided in the prompt.",
    # only score
    "Your task is to generate a summary based on the user's past reviews, in English, of the user's common scores given for positive and negative reviews. The summary will be used to answer only the last question. You should not provide the direct answer to the question. Instead, focus on identifying the key points for mapping the attitude of the user to the scores given for positive and negative reviews. Use the following format: The user's common scores given for positive and negative reviews are: 1. '[Top1 most common positive score: <Top1 most common positive score>]' 2. '[Top2 most common positive score: <Top2 most common positive score>]' ... 1. '[Top1 most common negative score: <Top1 most common negative score>]' 2. '[Top2 most common negative score: <Top2 most common negative score>]' ... Focus only on the last question provided in the prompt.",
    ]

# news headline generation, "generation"
lamp4_white_prompt_words = [
    "Given this author's previous articles, try to describe a template for their headlines. I want to be able to accurately predict the headline gives one of their articles. Be specific about their style and wording, don't tell me anything generic. Use the following format: The template is: '[template 1]', '[template 2]', '[template 3]', '[template 4]'.",
    "Your task is to generate a template based on the author's previous news articles, in English, that reflects the author's wording and style. The template should be able to accurately predict the headline given one of their articles, so that can be used to answer the last question. Use the following format: The template is: '[template 1]', '[template 2]', '[template 3]', '[template 4]', '[template 5]'."
    ]


lamp1_black_prompt_words = [
    # with rag
    "Answer the question about the author's reference choice based on the summary of the author's research interests and the author's relevant papers published previously. Only output the answer, do not provide any explanation.",
    # without rag
    "Answer the question about the author's reference choice based on the summary of the author's research interests. Only output the answer, do not provide any explanation.",
    # only rag
    "Answer the question about the author's reference choice based on the author's papers published previously. Only output the answer, do not provide any explanation."
]


lamp2M_black_prompt_words = [
    # with rag
    "Answer the question about the tag for a movie based on the most popular category and some relevant movie, tag pairs. Only output the answer, do not provide any explanation.",
    # without rag
    "Answer the question about the tag for a movie based on the most popular category. Only output the answer, do not provide any explanation.",
    # only rag
    "Answer the question about the tag for a movie based on some movies and their tags. Only output the answer, do not provide any explanation."
]

# to be updated
lamp2N_black_prompt_words = [
    # with rag
    "Answer the question based on the most popular category and some relevant article, category pairs. Only output the answer, do not provide any explanation.",
    # without rag
    "Answer the question based on the most popular category. Only output the answer, do not provide any explanation.",
    # only rag 
    "Answer the question based on some relevant articles and their categories. Only output the answer, do not provide any explanation."
]

lamp3_black_prompt_words = [
    # with rag
    "Answer the question about the user's rating score based on the summary of user's common scores given for positive and negative reviews and the user's relevant past reviews. Only output the answer, do not provide any explanation.",
    # without rag
    "Answer the question based on the summary of the user's common scores given for positive and negative reviews. Only output the answer, do not provide any explanation.",
    # only rag
    "Answer the question based on the user's past reviews and scores. Only output the answer, do not provide any explanation."
]

lamp4_black_prompt_words = [
    # with rag
    "Generate a news headline for a given article based on the template of the author's previous news articles and the author's relevant past articles. Only output the headline, do not provide any explanation.",
    # without rag
    "Generate a news headline for a given article based on the template of the author's previous news articles. Only output the headline, do not provide any explanation.",
    # only rag
    "Generate a news headline for a given article based on the author's past articles and headlines. Only output the headline, do not provide any explanation."
]