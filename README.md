# Nemo
Guiding and Contextualizing Weak Supervision for Interactive Data Programming

## Environment Setup
`pip install -r requirements.txt`

## Example Run
- Random selection: `python interactive_dp.py --dataset AmazonReview --query-method random`
- SEU selection: `python interactive_dp.py --dataset AmazonReview --query-method uncertainty_lm --seu`
