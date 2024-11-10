from rank_bm25 import BM25Okapi
from prompt.utils import extract_strings_between_quotes, extract_after_article, extract_after_review, extract_after_description
from prompt.lamp_format import lamp1_white_prompt_words, lamp3_white_prompt_words, lamp4_white_prompt_words, lamp1_black_prompt_words, lamp3_black_prompt_words, lamp4_black_prompt_words  
from prompt.lamp_format import lamp2M_white_prompt_words, lamp2N_white_prompt_words, lamp2M_black_prompt_words, lamp2N_black_prompt_words

import random
import json
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")    
HF_TOKEN="your_token"


# ============================== retrieval ==============================
# lamp1
def classification_citation_query_corpus_maker(inp, profile):
    corpus = [f'{x["title"]} {x["abstract"]}' for x in profile]
    extracted = extract_strings_between_quotes(inp)
    query = f'{extracted[1]} {extracted[2]}'
    return corpus, query

# lamp2n
def classification_news_query_corpus_maker(inp, profile):
    corpus = [f'{x["title"]} {x["text"]}' for x in profile]
    query = extract_after_article(inp)
    return corpus, query

# lamp2m
def classification_movies_query_corpus_maker(inp, profile):
    corpus = [f'{x["description"]}' for x in profile]
    query = extract_after_description(inp)
    return corpus, query

# lamp3
def classification_review_query_corpus_maker(inp, profile):
    corpus = [f'{x["text"]}' for x in profile]
    query = extract_after_review(inp)
    return corpus, query

# lamp4
def generation_news_query_corpus_maker(inp, profile):
    corpus = [f'{x["title"]} {x["text"]}' for x in profile]
    query = extract_after_article(inp)
    return corpus, query


# ============================== summary prompts for whitebox ==============================

def lamp1_summary_prompt(profile, question, prompt):
    paper_titles = []
    for p in profile:
        paper_titles.append(f'"{p["title"]}"')
    final_prompt = prompt + " " + "The published papers are: " + ", and ".join(paper_titles) + "."
    return final_prompt

def lamp2n_summary_prompt(profile, question, prompt):
    news = []
    for p in profile:
        text_news = f'the category for the article: "{p["text"]}" is "{p["category"]}" '
        news.append(text_news)
    return prompt + " " + "The articles and categories are: " + ", and ".join(news) + "."

def lamp2m_summary_prompt(profile, question, prompt):
    movies = []
    for p in profile:
        text_movies = f'the tag for the movie: "{p["description"]}" is "{p["tag"]}" '
        movies.append(text_movies)
    return prompt + " " + "The movies and tags are: " + ", and ".join(movies) + "."

def lamp3_summary_prompt(profile, question, prompt):
    reviews = []
    for p in profile:
        text = f'{p["score"]} is the score for review "{p["text"]}"'
        reviews.append(text)
    return prompt + " " + "The scores and reviews are: " + ", and ".join(reviews) + "."

def lamp4_summary_prompt(profile, question, prompt):
    articles = []
    for p in profile:
        text = f'"{p["title"]}" is the title for "{p["text"]}"'
        articles.append(text)
    return prompt + " " + "Previous articles and titles are: " + ", and ".join(articles) + "."


# ============================== black prompts pag ==============================

def lamp1_black_prompt(question, summary, profile, prompt, is_rag):
    if is_rag:
        paper_titles = []
        for p in profile:
            paper_titles.append(f'"{p["title"]}"')
        return prompt + " " + "Summary: " + summary + " " + "The published papers are: " + ", and ".join(paper_titles) + "." + " The question is: " + question + " " + "Only output the answer, do not provide any explanation!!"
    else:
        return prompt + " " + "Summary: " + summary + " " + "The question is: " + question + " " + "Only output the answer, do not provide any explanation!!"

def lamp2n_black_prompt(question, summary, profile, prompt, is_rag):
    if is_rag:
        news = []
        for p in profile:
            news.append(f'the category for the article: "{p["text"]}" is "{p["category"]}" ')
        return prompt + " " + "Summary: " + summary + " " + "The articles are: " + ", and ".join(news) + "." + " The question is: " + question + " " + "Only output the answer, do not provide any explanation!!"
    else:
        return prompt + " " + "Summary: " + summary + " " + "The question is: " + question + " " + "Only output the answer, do not provide any explanation!!"
    
def lamp2m_black_prompt(question, summary, profile, prompt, is_rag):
    if is_rag:
        movies = []
        for p in profile:
            movies.append(f'the tag for the movie: "{p["description"]}" is "{p["tag"]}" ')
        return prompt + " " + "Summary: " + summary + " " + "The movies are: " + ", and ".join(movies) + "." + " The question is: " + question + " " + "Only output the answer, do not provide any explanation!!"
    else:
        return prompt + " " + "Summary: " + summary + " " + "The question is: " + question + " " + "Only output the answer, do not provide any explanation!!"

def lamp3_black_prompt(question, summary, profile, prompt, is_rag):
    if is_rag:
        reviews = []
        for p in profile:
            review = f'{p["score"]} is the score for review "{p["text"]}" '
            reviews.append(review)
        return prompt + " " + "Summary: " + summary + " " + "Scores and reviews: " + ", and ".join(reviews) + "." + " The question is: " + question + " " + "Only output the score, do not provide any explanation!!"
    else:
        return prompt + " " + "Summary: " + summary + " " + "The question is: " + question + " " + "Only output the score, do not provide any explanation!!"


def lamp4_black_prompt(question, summary, profile, prompt, is_rag):
    if is_rag:
        articles = []
        for p in profile:
            article = f'"{p["title"]}" is the title for "{p["text"]}" '
            articles.append(article)
        return prompt + " " + "Template: " + summary + " " + "Articles: " + ", and ".join(articles) + "." + " The question is: " + question + " " + "Only output the headline, do not provide any explanation!!"
    else:
        return prompt + " " + "Template: " + summary + " " + "The question is: " + question + " " + "Only output the headline, do not provide any explanation!!"


# ============================== black prompts rag ==============================

def lamp1_black_prompt_rag(question, profile, prompt):
    paper_titles = []
    for p in profile:
        paper_titles.append(f'"{p["title"]}"')
    return prompt + " " + "The published papers are: " + ", and ".join(paper_titles) + "." + " The question is: " + question + " " + "Only output the answer, do not provide any explanation!!"

def lamp2n_black_prompt_rag(question, profile, prompt):
    news = []
    for p in profile:
        news.append(f'the category for the article: "{p["text"]}" is "{p["category"]}" ')
    return prompt + " " + "The articles are: " + ", and ".join(news) + "." + " The question is: " + question + " " + "Only output the answer, do not provide any explanation!!"

def lamp2m_black_prompt_rag(question, profile, prompt):
    movies = []
    for p in profile:
        movies.append(f'the tag for the movie: "{p["description"]}" is "{p["tag"]}" ')
    return prompt + " " + "The movies are: " + ", and ".join(movies) + "." + " The question is: " + question + " " + "Only output the answer, do not provide any explanation!!"

def lamp3_black_prompt_rag(question, profile, prompt):
    reviews = []
    for p in profile:
        review = f'{p["score"]} is the score for review "{p["text"]}" '
        reviews.append(review)
    return prompt + " " + "Scores and reviews: " + ", and ".join(reviews) + "." + " The question is: " + question + " " + "Only output the score, do not provide any explanation!!"


def lamp4_black_prompt_rag(question, profile, prompt):
    articles = []
    for p in profile:
        article = f'"{p["title"]}" is the title for "{p["text"]}" '
        articles.append(article)
    return prompt + " " + "Articles: " + ", and ".join(articles) + "." + " The question is: " + question + " " + "Only output the headline, do not provide any explanation!!"


def create_white_prompts(question, num_retrieve, is_ranked = False, use_all = False):
    def prompt(inp, profile, task):
        if task == "LaMP-1":
            corpus, query = classification_citation_query_corpus_maker(inp, profile)
        elif task == "LaMP-2M":
            corpus, query = classification_movies_query_corpus_maker(inp, profile)
        elif task == "LaMP-2N":
            corpus, query = classification_news_query_corpus_maker(inp, profile)
        elif task == "LaMP-3":
            corpus, query = classification_review_query_corpus_maker(inp, profile)
        elif task == "LaMP-4":
            corpus, query = generation_news_query_corpus_maker(inp, profile)
        
        num_profile = min(num_retrieve, len(profile))
        if not is_ranked: 
            tokenized_corpus = [x.split() for x in corpus]
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = query.split()
            selected_profs = bm25.get_top_n(tokenized_query, profile, n=num_profile)
        else:
            selected_profs_cont = profile[:num_profile] if not use_all else profile
            selected_profs = selected_profs_cont


        while True:
            try:
                if task == "LaMP-1":
                    prompt = lamp1_summary_prompt(selected_profs, question, lamp1_white_prompt_words[0])
                elif task == "LaMP-2M":
                    prompt = lamp2m_summary_prompt(selected_profs, question, lamp2M_white_prompt_words[0])
                elif task == "LaMP-2N":
                    prompt = lamp2n_summary_prompt(selected_profs, question, lamp2N_white_prompt_words[0])
                elif task == "LaMP-3":
                    prompt = lamp3_summary_prompt(selected_profs, question, lamp3_white_prompt_words[0])
                elif task == "LaMP-4":
                    prompt = lamp4_summary_prompt(selected_profs, question, lamp4_white_prompt_words[0])
                
                tokens = tokenizer.encode(prompt)
                if task == "LaMP-1":
                    if len(tokens) <= 4000:
                        return prompt
                    else:
                        # print("reduce num_retrieve of profiles")
                        selected_profs.pop(-2)
                        if not selected_profs:
                            return "Error: Not enough context to generate a valid prompt within token limit"
                else:
                    if len(tokens) <= 4000: 
                        return prompt
                    else:
                        # print("reduce num_retrieve of profiles")
                        if task == "LaMP-3":
                            selected_profs.pop(-1)
                        else:
                            selected_profs.pop(-2)
                        if not selected_profs:
                            return "Error: Not enough context to generate a valid prompt within token limit"
            except Exception as e:
                print(e)
                return "Error"
    return prompt


def create_black_prompts(num_retrieve, is_ranked = False, use_all = False, is_rag = False):
    def prompt(inp, profile, task, summary):
        if task == "LaMP-1":
            corpus, query = classification_citation_query_corpus_maker(inp, profile)
        elif task == "LaMP-2M":
            corpus, query = classification_movies_query_corpus_maker(inp, profile)
        elif task == "LaMP-2N":
            corpus, query = classification_news_query_corpus_maker(inp, profile)
        elif task == "LaMP-3":
            corpus, query = classification_review_query_corpus_maker(inp, profile)
        elif task == "LaMP-4":
            corpus, query = generation_news_query_corpus_maker(inp, profile)

        num_profile = min(num_retrieve, len(profile))
        if not is_ranked: 
            tokenized_corpus = [x.split() for x in corpus]
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = query.split()
            selected_profs = bm25.get_top_n(tokenized_query, profile, n=num_profile)
        
        else:
            selected_profs_cont = profile[:num_profile] if not use_all else profile
            selected_profs = selected_profs_cont


        if task == "LaMP-1":
            return lamp1_black_prompt(inp, summary, selected_profs, lamp1_black_prompt_words[1], is_rag)
            #return create_classification_citation_summary(selected_profs) 
        elif task == "LaMP-2M":
            return lamp2m_black_prompt(inp, summary, selected_profs, lamp2M_black_prompt_words[1], is_rag)
        elif task == "LaMP-2N":
            return lamp2n_black_prompt(inp, summary, selected_profs, lamp2N_black_prompt_words[1], is_rag)
        elif task == "LaMP-3":
            # return create_classification_review_summary(selected_profs)
            return lamp3_black_prompt(inp, summary, selected_profs, lamp3_black_prompt_words[1], is_rag)
        elif task == "LaMP-4":
            # return create_generation_news_summary(selected_profs)
            return lamp4_black_prompt(inp, summary, selected_profs, lamp4_black_prompt_words[1], is_rag)
        else:
            return "Error"

    return prompt

def create_black_prompts_rag(num_retrieve, is_ranked = False, use_all = False):
    def prompt(inp, profile, task):
        if task == "LaMP-1":
            corpus, query = classification_citation_query_corpus_maker(inp, profile)
        elif task == "LaMP-2M":
            corpus, query = classification_movies_query_corpus_maker(inp, profile)  
        elif task == "LaMP-2N":
            corpus, query = classification_news_query_corpus_maker(inp, profile)
        elif task == "LaMP-3":
            corpus, query = classification_review_query_corpus_maker(inp, profile)
        elif task == "LaMP-4":
            corpus, query = generation_news_query_corpus_maker(inp, profile)

        num_profile = min(num_retrieve, len(profile))
        if not is_ranked: 
            tokenized_corpus = [x.split() for x in corpus]
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = query.split()
            selected_profs = bm25.get_top_n(tokenized_query, profile, n=num_profile)
        
        else:
            selected_profs_cont = profile[:num_profile] if not use_all else profile
            selected_profs = selected_profs_cont


        if task == "LaMP-1":
            return lamp1_black_prompt_rag(inp, selected_profs, lamp1_black_prompt_words[2])
        elif task == "LaMP-2M":
            return lamp2m_black_prompt_rag(inp, selected_profs, lamp2M_black_prompt_words[2])
        elif task == "LaMP-2N":
            return lamp2n_black_prompt_rag(inp, selected_profs, lamp2N_black_prompt_words[2])
        elif task == "LaMP-3":
            # return create_classification_review_summary(selected_profs)
            return lamp3_black_prompt_rag(inp, selected_profs, lamp3_black_prompt_words[2])
        elif task == "LaMP-4":
            # return create_generation_news_summary(selected_profs)
            return lamp4_black_prompt_rag(inp, selected_profs, lamp4_black_prompt_words[2])
        else:
            return "Error"

    return prompt
