# MDAQA

Pytorch Implementation of the AAAI Paper: 

![](img/model.png)

Models trained on real-world data often mirror and exacerbate existing social biases. Traditional methods for mitigating these biases typically require prior knowledge of the specific biases to be addressed, and the social groups associated with each instance. In this paper, we introduce a novel adversarial training strategy that operates withour relying on prior bias-type knowledge (e.g., gender or racial bias) and protected attribute labels. Our approach dynamically identifies biases during model training by utilizing auxiliary bias detector. These detected biases are simultaneously mitigated through adversarial training. Crucially, we implement these bias detectors at various levels of the feature maps of the main model, enabling the detection of a broader and more nuanced range of bias features. Through experiments on racial and gender biases in sentiment and occupation classification tasks, our method effectively reduces social biases without the need for demographic annotations. Moreover, our approach not only matches but often surpasses the efficacy of methods that require detailed demographic insights, marking a significant advancement in bias mitigation techniques.

## Environment Setup

You need to clone our project

```
https://github.com/maxwellyin/MABR.git
```

Create the environment and download the packages

```
conda create -n MABR python==3.10
conda activate MABR
pip install -r requirements.txt
```

## Data Preparation

### Occupation Classification

You will need to get the dataset following the instructions from the original paper,
[Bias in Bios: A Case Study of Semantic Representation Bias in a High-Stakes Setting](https://arxiv.org/abs/1901.09451).
(Or write to me - as of the time of writing these lines I have a copy of the dataset).

### Sentiment Classification

You'll need to first get the dataset from [ELazar and Goldberg](https://github.com/yanaiela/demog-text-removal/blob/master/src/data/README.md) and have it at ``../data/moji/sentiment_race/``, for instance: ``../data/moji/sentiment_race/neg_neg``.

## Training and Evaluation

Please run the scripts in the order they are indexed.

## Citing 

If you found this repository is helpful, please cite our paper:
```

```