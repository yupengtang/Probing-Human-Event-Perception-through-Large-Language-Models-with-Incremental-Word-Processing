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
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn 
import statsmodels.api as sm

os.chdir('C:/Users/Hanbo/Documents/GitHub/LLM_event_segmentation')

def read_and_clean_csv(file_path, num_cols=9):
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
    
def save_load_dataset(dataset_name,model,context_length = 'unlimited', predictions_data = None, mode = 'save'):
    if context_length == 'unlimited':
        datasets_paths = {
            'monkey': {
            'density': 'Dataset/button_press_proportion/monkey_button_density.csv',
            'predictions': f'Dataset/extract-embeddings-data/results/monkey/{model}_predictions.csv'
        },
        'pieman': {
            'density': 'Dataset/button_press_proportion/pieman_button_density.csv',
                'predictions': f'Dataset/extract-embeddings-data/results/pieman/{model}_predictions.csv'
            },
            'tunnel': {
                'density': 'Dataset/button_press_proportion/tunnel_button_density.csv',
            'predictions': f'Dataset/extract-embeddings-data/results/tunnel/{model}_predictions.csv'
            }
        }
    else:
        datasets_paths = {
            'monkey': {
            'density': 'Dataset/button_press_proportion/monkey_button_density.csv',
            'predictions': f'Dataset/extract-embeddings-data/results/monkey/{model}_predictions_{context_length}.csv'
        },
        'pieman': {
            'density': 'Dataset/button_press_proportion/pieman_button_density.csv',
                'predictions': f'Dataset/extract-embeddings-data/results/pieman/{model}_predictions_{context_length}.csv'
            },
            'tunnel': {
                'density': 'Dataset/button_press_proportion/tunnel_button_density.csv',
            'predictions': f'Dataset/extract-embeddings-data/results/tunnel/{model}_predictions_{context_length}.csv'
            }
        }

    paths = datasets_paths[dataset_name]
    if mode == 'save' and predictions_data is not None:
        # Save predictions and density data
        predictions_data.to_csv(paths['predictions'], index=False)
    elif mode == 'load':
        # Load predictions with encoding correction
        predictions_data = read_and_clean_csv(paths['predictions'])  # Fix UnicodeDecodeError
    else:
        raise ValueError("Invalid mode or missing predictions data")
    density_data = pd.read_csv(paths['density'])
    return predictions_data, density_data

# Store results for visualization and analysis
def prepare_data_for_analysis(model,context_length = 'unlimited', load_only = True):
    final_results = {}

    # load word frequency norm data
    # Reading the Excel file
    if not load_only:
        word_frequency_norm_data = pd.read_excel('Dataset/subtlexus_norm.xlsx')

    for dataset in ['monkey', 'pieman', 'tunnel']:
        predictions_data, density_data = save_load_dataset(dataset, model,context_length, predictions_data = None, mode = 'load')
        
        if not load_only:
            # collect Lg10WF and Lg10CD
            for word in predictions_data['token']:
                # if the word is not in the word_frequency_norm_data, use nan to fill
                if word not in word_frequency_norm_data['Word'].values:
                    word_frequency_norm = np.nan
                    word_context_norm = np.nan
                else:
                    word_frequency_norm = word_frequency_norm_data[word_frequency_norm_data['Word'] == word]['Lg10WF'].values[0]
                    word_context_norm = word_frequency_norm_data[word_frequency_norm_data['Word'] == word]['Lg10CD'].values[0]
                predictions_data.loc[predictions_data['token'] == word, 'Lg10WF'] = word_frequency_norm
                predictions_data.loc[predictions_data['token'] == word, 'Lg10CD'] = word_context_norm
            save_load_dataset(dataset, model,context_length, predictions_data = predictions_data, mode = 'save')
            
        # Normalize density and calculate Bayesian Surprise
        density_data.columns = ['time', 'density']
        resampled_density = np.interp(
            np.linspace(0, len(density_data['time']) - 1, len(predictions_data)),
            np.arange(len(density_data['time'])),
            density_data['density']
        )
        normalized_density = (resampled_density - resampled_density.min()) / (resampled_density.max() - resampled_density.min())
        # convert likelihood_YES and likelihood_NO to float
        predictions_data['likelihood_YES'] = predictions_data['likelihood_YES'].astype(float)
        predictions_data['likelihood_NO'] = predictions_data['likelihood_NO'].astype(float)
        # convert Lg10WF and Lg10CD non-NA values to float
        predictions_data['Lg10WF'] = pd.to_numeric(predictions_data['Lg10WF'], errors='coerce')
        predictions_data['Lg10CD'] = pd.to_numeric(predictions_data['Lg10CD'], errors='coerce')
        #normalize the likelihood of 'YES' and 'NO'
        # if no normalization
        normalized_likelihood_YES = (predictions_data['likelihood_YES'] - predictions_data['likelihood_YES'].min()) / \
                            (predictions_data['likelihood_YES'].max() - predictions_data['likelihood_YES'].min())
        normalized_likelihood_NO = (predictions_data['likelihood_NO'] - predictions_data['likelihood_NO'].min()) / \
                            (predictions_data['likelihood_NO'].max() - predictions_data['likelihood_NO'].min())
        
        # normalize Lg10WF and Lg10CD
        normalized_Lg10WF = (predictions_data['Lg10WF'] - predictions_data['Lg10WF'].min()) / \
                            (predictions_data['Lg10WF'].max() - predictions_data['Lg10WF'].min())
        normalized_Lg10CD = (predictions_data['Lg10CD'] - predictions_data['Lg10CD'].min()) / \
                           (predictions_data['Lg10CD'].max() - predictions_data['Lg10CD'].min())
        
        # Store results
        final_results[dataset] = {
            'normalized_density': normalized_density,
            'normalized_likelihood_YES': normalized_likelihood_YES,
            'normalized_likelihood_NO': normalized_likelihood_NO,
            'likelihood_YES': predictions_data['likelihood_YES'],
            'likelihood_NO': predictions_data['likelihood_NO'],
            'Lg10WF': normalized_Lg10WF,
            'Lg10CD': normalized_Lg10CD
        }

    return final_results

model = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
context_length = 64
final_results = prepare_data_for_analysis(model,context_length)

for dataset, result in final_results.items():
    plt.figure(figsize=(10, 6))
    plt.plot(result['normalized_density'], label=f'{dataset.capitalize()} Human Event Density', color='blue')
    plt.plot(result['normalized_likelihood_YES'], label=f'{dataset.capitalize()} Likelihood YES', color='orange', alpha=0.7)
    plt.title(f"Likelihood vs Human Event Density ({dataset.capitalize()})")
    plt.xlabel("Time (Aligned Index)")
    plt.ylabel("Normalized Values")
    plt.legend()
    plt.show()




def plot_correlation(model, context_length, max_lag = 20):
    # prepare data for analysis
    final_results = prepare_data_for_analysis(model,context_length)

    for dataset, result in final_results.items():
    # Extract data
        density = pd.Series(result['normalized_density'])  # Convert to pandas Series
        likelihood_yes = pd.Series(result['normalized_likelihood_YES'])  # Convert to pandas Series
        
        # Compute correlations for each lag
        lags = range(-max_lag, max_lag + 1)
        lag_correlations = []
        
        for lag in lags:

            # Shift likelihood YES forward for positive lag
            shifted_likelihood_yes = likelihood_yes.shift(lag)
            
            # Drop NaNs caused by shifting
            valid_data = pd.DataFrame({
            'shifted_likelihood_yes': shifted_likelihood_yes,
            'density': density
            }).dropna()
            
            # Compute Pearson correlation
            correlation = valid_data['shifted_likelihood_yes'].corr(valid_data['density'])
            lag_correlations.append(correlation)
    
        # Plot original data
        plt.figure(figsize=(10, 6))
        plt.plot(density, label=f'{dataset.capitalize()} Human Event Density', color='blue')
        plt.plot(likelihood_yes, label=f'{dataset.capitalize()} Likelihood YES', color='orange', alpha=0.7)
        plt.title(f"Likelihood YES vs Human Event Density ({dataset.capitalize()})_{model}_{context_length}")
        plt.xlabel("Time (Aligned Index)")
        plt.ylabel("Normalized Values")
        plt.legend()
        save_path = f'pic/raw_data/{dataset}/{model}'
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}/{context_length}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Plot lag-based correlations
        plt.figure(figsize=(10, 6))
        plt.plot(lags, lag_correlations, label=f'{dataset.capitalize()} Lag Correlations', color='green')
        # plot the red with max correlation lag
        max_corr_lag = lags[np.argmax(lag_correlations)]
        plt.axvline(max_corr_lag, color='red', linestyle='--', label=f'Lag = {max_corr_lag}')
        plt.title(f"Lag-Based Correlation: Likelihood YES vs Human Density ({dataset.capitalize()})_{model}_{context_length}")
        plt.xlabel("Lag (Index Shift)")
        plt.ylabel("Pearson Correlation Coefficient")
        plt.legend()
        save_path = f'pic/lag_correlation/{dataset}/{model}'
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}/{context_length}.png', dpi=300, bbox_inches='tight')
        plt.close()



def collect_correlation(models, context_lengths, max_lag = 20, load_only = True, controlled = False):
    correlation_results = {}
    lags = range(-max_lag, max_lag + 1)
    for model in models:
        correlation_results[model] = {}
        for context_length in context_lengths:
            correlation_results[model][context_length] = {}
            #collect two correlation, one is lag 0, one is max lag
            final_results = prepare_data_for_analysis(model,context_length, load_only = load_only)
            for dataset, result in final_results.items():
                correlation_results[model][context_length][dataset] = {}
                density = pd.Series(result['normalized_density'])  # Convert to pandas Series
                likelihood_yes = pd.Series(result['normalized_likelihood_YES'])  # Convert to pandas Series
                Lg10WF = pd.Series(result['Lg10WF'])  # Convert to pandas Series
                Lg10CD = pd.Series(result['Lg10CD'])  # Convert to pandas Series
                lag_correlations = []
                for lag in lags:
                    shifted_likelihood_yes = likelihood_yes.shift(lag)
                    shifted_Lg10WF = Lg10WF.shift(lag)  
                    shifted_Lg10CD = Lg10CD.shift(lag)
                    valid_data = pd.DataFrame({
                    'shifted_likelihood_yes': shifted_likelihood_yes,
                    'density': density,
                    'shifted_Lg10WF': shifted_Lg10WF,
                    'shifted_Lg10CD': shifted_Lg10CD
                    }).dropna()
                    # use linear prediction to control for Lg10WF and Lg10CD
                    # linear prediction
                    if controlled:
                        X = valid_data[['shifted_likelihood_yes', 'shifted_Lg10WF', 'shifted_Lg10CD']]
                    else:
                        X = valid_data[['shifted_likelihood_yes']]

                    y = valid_data['density']
                    # add a constant term to the X
                    X = sm.add_constant(X)
                    linear_model = sm.OLS(y, X)
                    results = linear_model.fit()
                    
                    # Get goodness of fit metrics
                    beta_shifted_likelihood_yes = results.params['shifted_likelihood_yes']
                    r_squared = results.rsquared  # R-squared value
                    adj_r_squared = results.rsquared_adj  # Adjusted R-squared
                    f_stat = results.fvalue  # F-statistic
                    f_pvalue = results.f_pvalue  # P-value for F-test
                    aic = results.aic  # Akaike Information Criterion
                    bic = results.bic  # Bayesian Information Criterion
                    
                    # You can store these metrics in your correlation_results dictionary
                    lag_correlations.append({
                        'beta': beta_shifted_likelihood_yes,
                        'r_squared': r_squared,
                        'adj_r_squared': adj_r_squared,
                        'f_stat': f_stat,
                        'f_pvalue': f_pvalue,
                        'aic': aic,
                        'bic': bic})
                
                # Convert list of dicts to dict of lists
                lag_correlations = {
                    key: [d[key] for d in lag_correlations]
                    for key in lag_correlations[0].keys()
                }
                
                # find the max correlation lag
                max_corr_lag = lags[np.argmax(lag_correlations['beta'])]
                correlation_results[model][context_length][dataset]['max_corr'] = {'lag': max_corr_lag, 'correlation': lag_correlations['beta'][max_corr_lag],'metrics':lag_correlations['adj_r_squared'][max_corr_lag]}
                correlation_results[model][context_length][dataset]['lag_0_corr'] = {'lag': 0, 'correlation': lag_correlations['beta'][0],'metrics':lag_correlations['adj_r_squared'][0]}
    return correlation_results

models = ['gpt-4o-2024-08-06', 'meta-llama/Meta-Llama-3.1-8B-Instruct']
context_lengths = [64, 128, 256, 512, 1024, 'unlimited']
correlation_results = collect_correlation(models, context_lengths, max_lag = 50, controlled = False)
controlled_correlation_results = collect_correlation(models, context_lengths, max_lag = 50, controlled = True)

for model in models:
    for context_length in context_lengths:
        plot_correlation(model, context_length, max_lag = 50)

def plot_max_correlations_by_task(correlation_results, models, context_lengths, tasks=['monkey', 'pieman', 'tunnel'], controlled = False):
    """
    Create grouped bar plots for max correlations by task.
    
    Args:
    - correlation_results: Dictionary containing correlation results
    - models: List of model names
    - context_lengths: List of context lengths
    - tasks: List of task names
    """
    
    # Set style parameters
    plt.style.use('default')  # Using default style instead of seaborn
    bar_width = 0.15
    colors = plt.cm.viridis(np.linspace(0, 1, len(context_lengths)))
    
 
    
    display_names = [model for model in models]
    
    for task in tasks:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate x positions for bars
        x = np.arange(len(models))
        
        # Plot bars for each context length
        for i, context_length in enumerate(context_lengths):
            correlations = [correlation_results[model][context_length][task]['max_corr']['correlation'] 
                          for model in models]
            
            ax.bar(x + i*bar_width, 
                  correlations,
                  bar_width,
                  label=str(context_length),
                  color=colors[i],
                  alpha=0.8)
        
        # Customize plot
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Maximum Correlation', fontsize=12)
        ax.set_title(f'{task.capitalize()} Task', fontsize=14)
        
        # # add metrcis as text on the top of the bar
        # for i, context_length in enumerate(context_lengths):
        #     for j, model in enumerate(models):
        #         metrics = correlation_results[model][context_length][task]['max_corr']['metrics']
        #         ax.text(x[j] + i*bar_width, correlations[j], 'Adjusted R^2: ' + f'{metrics:.2f}', ha='center', va='bottom')
                
        # Set x-ticks at the center of each group
        ax.set_xticks(x + bar_width * (len(context_lengths)-1)/2)
        ax.set_xticklabels(display_names, 
                          rotation=0,
                          ha='center',
                          fontsize=12)
        
        # Add legend with title
        ax.legend(title='Context Length',
                 bbox_to_anchor=(1.05, 1),
                 loc='upper left',
                 fontsize=10,
                 title_fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Adjust layout
        plt.tight_layout()
        
        # # Save figure
        if controlled:
            save_path = f'pic/controlled_max_correlation_{task}'
        else:
            save_path = f'pic/max_correlation_{task}'
        plt.savefig(f'{save_path}.png', 
                   dpi=300, 
                   bbox_inches='tight')
        plt.show()

# Usage:
plot_max_correlations_by_task(correlation_results, models, context_lengths)
plot_max_correlations_by_task(controlled_correlation_results, models, context_lengths, controlled = True)







