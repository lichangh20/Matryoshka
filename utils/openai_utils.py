def check_cost(prompt_tok, completion_tok, model_type = 'gpt-3.5-turbo'):
    prompt_tok /= 1000
    completion_tok /= 1000
    if model_type == 'gpt-3.5-turbo':
        return prompt_tok*0.001 + completion_tok*0.002
    elif model_type == 'gpt-4':
        return prompt_tok*0.03 + completion_tok*0.06
    elif model_type == 'gpt-4o':
        return prompt_tok*0.005 + completion_tok*0.015
    elif model_type == 'gpt-4o-mini':
        return prompt_tok*0.00015 + completion_tok*0.0006