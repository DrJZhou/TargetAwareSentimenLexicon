# Target-Aware Sentimen Lexicon
Codes for our EACL paper &lt;Is “hot pizza” Positive or Negative? Mining Target-aware Sentiment Lexicons>


# Requirements
- Python = 3.6
- Pytorch = 1.4.0
- Transformers = 2.8.0

# Distant Supervision
export GLUE_DIR=.
export TASK_NAME=sentiment_res

python DistantSupervision.py --evaluate_during_training --overwrite_output_dir --local_rank -1 --model_type bert --model_name_or_path bert-base-uncased --task_name $TASK_NAME --do_train --do_eval --do_lower_case --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length 500 --per_gpu_eval_batch_size 8 --per_gpu_train_batch_size 8 --learning_rate 2e-5 --num_train_epochs 5.0 --output_dir ./tmp_res/$TASK_NAME/ > log_.out


# Discrete/Continuous Perturbation
python DiscretePerturbation.py
python ContinuousPerturbation.py

# Lexicon
python Obtain_TopK.py

## Cite
If you find our code is useful, please cite:
```
@inproceedings{zhou-etal-2021-hot,
    title = "Is {``}hot pizza{''} Positive or Negative? Mining Target-aware Sentiment Lexicons",
    author = "Zhou, Jie  and
      Wu, Yuanbin  and
      Sun, Changzhi  and
      He, Liang",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.eacl-main.49",
    pages = "608--618",
    abstract = "Modelling a word{'}s polarity in different contexts is a key task in sentiment analysis. Previous works mainly focus on domain dependencies, and assume words{'} sentiments are invariant within a specific domain. In this paper, we relax this assumption by binding a word{'}s sentiment to its collocation words instead of domain labels. This finer view of sentiment contexts is particularly useful for identifying commonsense sentiments expressed in neural words such as {``}big{''} and {``}long{''}. Given a target (e.g., an aspect), we propose an effective {``}perturb-and-see{''} method to extract sentiment words modifying it from large-scale datasets. The reliability of the obtained target-aware sentiment lexicons is extensively evaluated both manually and automatically. We also show that a simple application of the lexicon is able to achieve highly competitive performances on the unsupervised opinion relation extraction task.",
}
```
