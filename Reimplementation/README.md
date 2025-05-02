# Investigating LLMs in Incremental Sentence Processing

## Overview
This project explores whether Large Language Models (LLMs) can replicate human incremental sentence processing behaviors. Specifically, it examines the ability of GPT-derived metrics to predict human event segmentation during narrative comprehension, using Bayesian Surprise as a key measure.

### Key Objectives
1. Investigate the relationship between human event segmentation and prediction error.
2. Test whether Bayesian Surprise, derived from GPT-2, aligns with human-annotated event boundaries.
3. Compare human sentence processing behaviors under varying sentence structures and conditions.

## Data Sources
The study leverages human segmentation data and GPT-2-based predictions from three narrative datasets:
- **Monkey in the Middle** (newly acquired segmentation data)
- **Pieman**
- **Tunnel Under the World**

The datasets used in this project can be downloaded from the following Zenodo link:  
[**Zenodo Repository - Incremental Sentence Processing Datasets**](https://zenodo.org/records/8404102)  

These datasets include behavioral event boundaries identified by human participants as well as model predictions necessary for computing Bayesian Surprise metrics.

## Methodology
- **Human Segmentation Data**: Behavioral event boundary data was collected via human listeners identifying transitions in narratives.
- **Bayesian Surprise Computation**: GPT-2's predicted probabilities were used to calculate Bayesian Surprise (via Kullback–Leibler Divergence).
- **Statistical Analysis**: ANOVA and regression models were applied to compare Bayesian Surprise with human segmentation data.

## Key Findings
- Transient Bayesian Surprise effectively predicts human event segmentation.
- Comparisons across sentence conditions (e.g., Sentence, Embedding Short, Embedding Long) reveal distinct human-LM processing correlations.
- Results align with findings from [Kumar et al. (2023)] which demonstrated the utility of Bayesian Surprise in narrative comprehension.

## Dependencies
- Python 3.9+
- Libraries: `pandas`, `numpy`, `matplotlib`, `scipy`, `seaborn`, `sklearn`

## How to Run
1. Clone the repository.
2. Download the datasets from [Zenodo](https://zenodo.org/records/8404102) and place them in the appropriate data folder.
3. Install dependencies: `pip install -r requirements.txt`.
4. Run the Jupyter Notebook `LLMs_Incremental_Sentence_Processing.ipynb`.

## Reference
This project was inspired by the work of Kumar et al. (2023) in their research on Bayesian Surprise as a predictor of human event segmentation. Their study offers foundational insights into the computational modeling of human narrative comprehension. For more details, refer to the original paper:  
**Kumar et al., "Bayesian Surprise Predicts Human Event Segmentation in Story Listening" (2023).**

## License
This project is open-source and adheres to the MIT license.

---

**Note**: This work builds upon pre-existing human behavioral data combined with computational outputs. Please credit appropriately if using parts of this repository.
