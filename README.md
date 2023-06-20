# Nemo: Guiding and Contextualizing Weak Supervision for Interactive Data Programming
Code for paper [Nemo: Guiding and Contextualizing Weak Supervision for Interactive Data Programming](https://arxiv.org/abs/2203.01382) (VLDB 2023)


## Environment Setup
```
python3 -m venv nemo_venv
source nemo_venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip uninstall -y nvidia_cublas_cu11
```

## Download Data
```
gdown 1C48r6FCw-hU6ACbO9BMIgdtHkJn6AMB2
unzip nemo_data.zip && rm nemo_data.zip
```

## Example Command Usages
Under directory `src/`:
- Snorkel (select by random):
  ```
  python interactive_dp.py  --dataset AmazonReview  --label-model snorkel --soft-training --query-method random
  ```
- Snorkel-Abs (select by abstain):
  ```
  python interactive_dp.py  --dataset AmazonReview  --label-model snorkel --soft-training --query-method abstain
  ```
- Snorkel-Dis (select by disagreement):
  ```
  python interactive_dp.py  --dataset AmazonReview  --label-model snorkel --soft-training --query-method disagreement
  ```
- Nemo:
  ```
  python interactive_dp.py  --dataset AmazonReview  --label-model snorkel --soft-training --query-method uncertainty_lm --seu --aggregate weighted --discard grid
  ```


## Cite
If you find this repository useful, please consider citing:
```
@article{hsieh2022nemo,
  title={Nemo: Guiding and contextualizing weak supervision for interactive data programming},
  author={Hsieh, Cheng-Yu and Zhang, Jieyu and Ratner, Alexander},
  journal={arXiv preprint arXiv:2203.01382},
  year={2022}
}
```
