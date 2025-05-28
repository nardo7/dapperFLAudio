[//]: # ">📋  A template README.md for code accompanying a Machine Learning paper"

# Domain Adaptive Federated Learning for Speech Emotion Recognition

[//]: # ">📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials"
[//]: # ">📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials"
[//]: <img src="./docs/DapperFL_overview.jpg" width = "80%" height = "" alt="DapperFL" TITLE="Overview of DapperFL." />

## Requirements

To install requirements first install miniconda and afterwards:

```setup
conda env create -f environment.yml
```

We use <a href="https://wandb.ai/">wandb</a> to keep a log of our experiments.
If you don't have a wandb account, just install it and use it as offline mode.

### Datasets

#### Digits

In principle you do not need to download the datasets yourself. The only exception is syn which can be found [here](https://www.kaggle.com/datasets/prasunroy/synthetic-digits/data)

[//]: # ">📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc..."

#### SER

1. You need to download CREMA-D from Kaggle, put it into the ./data folder. You just have to make sure that there is the following folder structure in the project root path: `./CREMA-D/AudioWAV`.
2. Run `python orginize_datasets.py` script. This will partition randomly the dataset into the train and test path, which will be then used by the fl_ser dataset

## Training & Evaluation

To train the model(s) in the paper, run this command:

```train
python ./fedml_experiments/standalone/domain_generalization/main.py --model dapperfl --dataset fl_ser --backbone resnet10 --parti_num 72 --online_ratio 0.1 --pr_strategy iterative
```

[//]: # ">📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters."

## Arguments

You can modify the arguments to run DapperFL on other settings. The arguments are described as follows:
| Arguments | Description |
|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| prefix | A prefix for logging. |
| communication_epoch | Total communication rounds of Federated Learning. |
| local_epoch | Local epochs for local model updating. |
| parti_num | Number of participants. |
| model | Name of FL framework. |
| dataset | Datasets used in the experiment. Options: fl_officecaltech, fl_digits, fl_ser. |
| pr_strategy | Pruning ratio used to prune local models. Options: 0 (without pruning), 0.1 ~ 0.9, AD (adaptive pruning), iterative (iterative pruning). |
| backbone | Backbone global model. Options: resnet10, resnet18. |
| alpha | Coefficient alpha in co-pruning. Default: 0.9. |
| alpha_min | Coefficient alpha_min in co-pruning. Default: 0.1. |
| epsilon | Coefficient epsilon in co-pruning. Default: 0.2. |
| reg_coeff | Coefficient for L2 regularization. Default: 0.01. |
| seed | Random seed.

[//]: # ">📋 Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below)."
[//]: # "## Pre-trained Models"
[//]: #
[//]: # "You can download pretrained models here:"
[//]: #
[//]: # "- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. "
[//]: #
[//]: # ">📋 Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable). Alternatively you can have an additional column in your results table with a link to the models."
[//]: # "## Results"
[//]: #
[//]: # "Our model achieves the following performance on :"
[//]: #
[//]: # "### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)"
[//]: #
[//]: # "| Model name | Top 1 Accuracy | Top 5 Accuracy |"
[//]: # "| ------------------ |---------------- | -------------- |"
[//]: # "| My awesome model | 85% | 95% |"
[//]: #
[//]: # ">📋 Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. "
[//]: # "## Contributing"
[//]: #
[//]: # ">📋 Pick a licence and describe how to contribute to your code repository. "
