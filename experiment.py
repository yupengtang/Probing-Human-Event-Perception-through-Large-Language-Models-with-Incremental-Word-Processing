# Load model directly
import numpy as np
import pandas as pd
from ast import literal_eval
import math
from tqdm import tqdm
import transformers
import numpy as np
from torch import cuda, bfloat16
import torch
import os
import re
import torch.nn.functional as F
import csv
# import openai
from torch.nn.functional import log_softmax
from pathlib import Path
import argparse
import openai

def load_dataset(dataset_name):
    datasets_paths = {
        'monkey': {
            'density': 'Dataset/button_press_proportion/monkey_button_density.csv',
            'predictions': 'Dataset/extract-embeddings-data/results/monkey/predictions.csv'
        },
        'pieman': {
            'density': 'Dataset/button_press_proportion/pieman_button_density.csv',
            'predictions': 'Dataset/extract-embeddings-data/results/pieman/predictions.csv'
        },
        'tunnel': {
            'density': 'Dataset/button_press_proportion/tunnel_button_density.csv',
            'predictions': 'Dataset/extract-embeddings-data/results/tunnel/predictions.csv'
        }
    }

    paths = datasets_paths[dataset_name]
   # Load predictions with encoding correction
    predictions_data = read_and_clean_csv(paths['predictions'])  # Fix UnicodeDecodeError
    density_data = pd.read_csv(paths['density'])
    return predictions_data, density_data


def model_loading(model_id):
    API_key = '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'


    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    # set quantization configuration to load large model with less GPU memory
    # this requires the `bitsandbytes` library
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    # begin initializing HF items, need auth token for these
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        use_auth_token=API_key
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        token=API_key,
        torch_dtype = 'auto'
    )
    model.eval()
    print(f"Model loaded on {device}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        token=API_key
    )
    if tokenizer.pad_token_id is None:
      tokenizer.add_special_tokens({'pad_token': '[-1]'})
      model.config.pad_token_id = -1
    return model, tokenizer


def read_and_clean_csv(file_path, num_cols=4):
    """
    Read CSV file and clean by keeping only specified number of columns.
    
    Args:
        file_path (str): Path to the CSV file
        num_cols (int): Number of columns to keep (default: 4)
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with specified number of columns
    """
    cleaned_data = []
    try:
        with open(file_path, 'r', encoding='ISO-8859-1') as f:
            reader = csv.reader(f)
            headers = next(reader)
            cleaned_data.append(headers[:num_cols])
            
            for i, row in enumerate(reader, 1):
                if len(row) > num_cols:
                    print(f"Row {i} had {len(row)} columns, truncating to {num_cols}")
                    cleaned_data.append(row[:num_cols])
                else:
                    cleaned_data.append(row)
                    
        return pd.DataFrame(cleaned_data[1:], columns=cleaned_data[0])
    
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return None


def generate_event_boundary_prompt(current_context: str, current_word: str) -> str:
    """
    Generates a prompt to ask the LLM whether the current word marks an event boundary.
    
    Args:
        current_context (str): The context of the story so far, including all prior words.
        current_word (str): The current word to evaluate for an event boundary.
    
    Returns:
        str: The formatted prompt for the LLM.
    """
    prompt = f""" You are analyzing a story one word at a time to determine event boundaries. An event boundary occurs when the narrative transitions from one event to another. This could involve changes in action, goals, characters, or settings.

            Context so far:
            {current_context}

            Current word:
            {current_word}

            Question:
            Is this word a transition between two events, marking an event boundary? Answer 'Yes' or 'No'.
            """
    return prompt


def get_behavior(prompt, model='gpt-4o-2024-08-06',config=None, max_attempts=5, task=None):
    if 'gpt' in model:
        tmp_message = {"role": "user", "content": prompt}
        openai.api_key = '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' 
        for attempt in range(max_attempts):
            try:
                tmp_response = openai.chat.completions.create(
                    model=model,
                    messages=[tmp_message],
                    temperature=config['temperature'],
                    max_tokens=config['max_tokens'],
                    top_p=1,
                    frequency_penalty=1,
                    presence_penalty=1,
                    logprobs = config['logprobs'],
                    top_logprobs = config['top_logprobs'],
                    seed=2024
                )
                choice = tmp_response.choices[0].message.content.strip().upper()
                

                if choice not in ['YES', 'NO']:
                    choice_idx = -1
                else:
                    if choice == 'YES':
                        choice_idx = 0
                    else:
                        choice_idx = 1
                return (choice_idx, tmp_response)
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_attempts - 1:
                    print(f"Failed to get a valid choice after {max_attempts} attempts. Using default choice.")
                    choice_idx = -1
                return (choice_idx, None)
                
    elif 'llama' in model:
        device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(device)
        # Generate response

        with torch.no_grad():
            generate_ids = llama_model.generate( **inputs, 
                                                max_new_tokens=config['max_tokens'],
                                                temperature=max(config['temperature'], 1e-10), 
                                                output_scores=True
                                            )
        generated_tokens = generate_ids[:, -1:]  # Just get the last token
        choice = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        choice = choice.upper()
        
        if choice == ' YES':
            choice_idx = 0
        elif choice == ' NO':
            choice_idx = 1
        else:
            choice_idx = -1

        outputs = llama_model(**inputs)
        logits = outputs.logits[:, -1, :]
        logits = log_softmax(logits, dim=-1).detach().cpu().numpy()
        del inputs, outputs, logits
        torch.cuda.empty_cache()
        return (choice_idx, logits)

    else:
        raise ValueError("Unsupported model. This function only supports GPT or LLAMA models.")


def extract_likelihood(response, model):
    if 'gpt' in model:
        logprobs_dict = {item.token: item.logprob for item in response.choices[0].logprobs.content[0].top_logprobs}
        logprob_YES = logprobs_dict.get('Yes', None)
        logprob_NO = logprobs_dict.get('No', None)
    elif 'llama' in model:
        # use tokenzier to encode only 'YES', 'NO' to get their id, then use these id to find out the logprob in response['response']
        keys = [' Yes', ' No']
        # tokeninze keys separately and get their ids in a list
        ids = []
        for key in keys:
            ids.append(tokenizer(key, return_tensors='pt').input_ids[0].tolist()[1])
        logprobs_dict = {key: response[0,id] for key, id in zip(keys, ids)}
        logprob_YES = logprobs_dict.get(' Yes', None)
        logprob_NO = logprobs_dict.get(' No', None)
    else:
        raise ValueError("Unsupported model. This function only supports GPT or LLAMA models.")
    

    # convert to likelihood
    likelihood_YES = math.exp(logprob_YES)
    likelihood_NO = math.exp(logprob_NO)
    # use softmax to normalize
    total_likelihood = likelihood_YES + likelihood_NO
    likelihood_YES = likelihood_YES / total_likelihood
    likelihood_NO = likelihood_NO / total_likelihood
    return {'YES':likelihood_YES, 'NO':likelihood_NO}

def run_experiment(model):
    """
    Run experiment for specific model and temperature combination
    
    Parameters:
    model (str): Model name
    temperature (float): Temperature value
    save_path (str): Path to save results
    """
    config = {
        'model': model,
        'temperature': 0,
        'max_tokens': 1,
        'logprobs': True,
        'top_logprobs': 20
    }
    # os.chdir('C:/Users/Hanbo/Documents/GitHub/LLM_event_segmentation')

    if 'llama' in model:
        global llama_model, tokenizer
        llama_model, tokenizer = model_loading(model)
    
    
    for task in ['monkey', 'pieman', 'tunnel']:
        predictions_data, _ = load_dataset(task)
        for context_length in [64, 128, 256, 512, 1024, 9999]:
            save_path = f'Dataset/extract-embeddings-data/results/{task}/{model}_predictions_{context_length}.csv'
            # Load existing results if any
            save_path = Path(save_path).resolve()
            
            # Create result directory if it doesn't exist
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # remove none rows in predictions_data
            predictions_data = predictions_data[predictions_data['datum word'].notna()]
            result_data = predictions_data.copy()
            result_data['likelihood_YES'] = None
            result_data['likelihood_NO'] = None
            result_data['event_boundary'] = None
            current_context = ''
            for i, row in tqdm(predictions_data.iterrows(), total=len(predictions_data), desc=f'Processing {task} with context length {context_length}'):
                current_word = row['token']
                current_context += current_word + ' '
                # if the context is longer than context_length, truncate the context from the beginning
                if len(current_context) > context_length:
                    current_context = current_context[-context_length:]
                prompt = generate_event_boundary_prompt(current_context, current_word)
                choice,response = get_behavior(prompt, model, config)
                likelihood = extract_likelihood(response, model)
                result_data.loc[i, 'likelihood_YES'] = likelihood['YES']
                result_data.loc[i, 'likelihood_NO'] = likelihood['NO']
                result_data.loc[i, 'event_boundary'] = choice
            del prompt, choice, response, likelihood
            torch.cuda.empty_cache()
            result_data.to_csv(save_path, index=False)
            del result_data

def main():
    parser = argparse.ArgumentParser(description='Run LLM experiments')
    parser.add_argument('--model', type=str, required=True,
                      choices=['gpt-4o-2024-08-06', 'gpt-4-turbo-2024-04-09',
                               'gpt-4o-mini-2024-07-18',
                               'meta-llama/Meta-Llama-3.1-8B-Instruct',
                               'meta-llama/Meta-Llama-3.1-70B-Instruct'],
                      help='Model to use')
    # parser.add_argument('--task', type=str, required=True,
    #                   choices=['monkey', 'pieman', 'tunnel'],
    #                   help='Task to use')
    # parser.add_argument('--context_length', type=int, required=True,default=9999,
    #                   help='Context length')
    args = parser.parse_args()

    # Run experiment
    run_experiment(args.model)

if __name__ == "__main__":
    main()




