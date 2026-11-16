# BBC News Text Classification: Four Approaches Compared

One classification problem, four methods, benchmarked under identical conditions
(same split, same held-out test set) to compare accuracy against real practical
tradeoffs, training data required, and compute cost, rather than assuming the most
complex method automatically wins.

## Problem

Classify a BBC News article into one of five categories: business, entertainment,
politics, sport, or tech.

## Dataset

[BBC News Archive](https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive),
2,225 articles, originally from Greene & Cunningham (2006). Clean dataset: no
duplicates, no missing values, reasonably balanced across the five categories.
The source file is tab-separated, not comma-separated, noted directly on the
dataset page and handled accordingly.

## Method

Four approaches, trained and evaluated on the same 70/15/15 stratified split:

1. **TF-IDF + Logistic Regression**: unigrams and bigrams, 10,000 max features.
2. **Pretrained Word2Vec (averaged) + Logistic Regression**: Google's 300-dimensional
   vectors (trained on ~100 billion words), since training Word2Vec from scratch on
   only 1,557 documents would produce undertrained embeddings. Each document is
   represented as the mean of its recognized words' vectors.
3. **Fine-tuned DistilBERT**: full model fine-tuning, checkpointed every epoch and
   selected by validation loss rather than final epoch, since a preliminary run
   showed real overfitting risk on a dataset this small.
4. **Zero-shot LLM classification**: no training data at all. Llama 3.1 8B (via
   Groq) is prompted with only the category names and article text.

## Results

Evaluated on a held-out test set (334 articles, untouched during training):

| Method | Test Accuracy | Training data required | Compute |
|---|---|---|---|
| TF-IDF + Logistic Regression | **98.80%** | 1,557 labeled examples | Seconds, CPU |
| DistilBERT (fine-tuned) | 97.90% | 1,557 labeled examples | ~4-5 min, GPU |
| Word2Vec (averaged) + Logistic Regression | 92.81% | 1,557 labeled examples + pretrained vectors | Minutes, CPU |
| Zero-shot LLM (Llama 3.1 8B) | 91.02% | None | API calls only |

**Key finding, stated honestly rather than forced into a narrative:** the two
approaches that fit a decision boundary to this specific dataset (TF-IDF, DistilBERT)
outperformed the two relying on general-purpose representations without task-specific
fitting (Word2Vec averaging, zero-shot), regardless of model complexity. The simplest
method (TF-IDF) matched or beat the most complex one (DistilBERT); fine-tuning provided
no measurable benefit over a bag-of-words baseline on this dataset. A plausible reason:
BBC News categories are lexically well-separated, sport vocabulary barely overlaps with
tech vocabulary, a condition under which methods that reward exact term matching can
compete with, or beat, methods built for deeper semantic understanding.

## Practical tradeoffs beyond accuracy

- **TF-IDF** is by far the cheapest and fastest to train and deploy, no GPU, no
  pretrained downloads, seconds not minutes.
- **Zero-shot** requires no labeled training data at all, a real advantage in a
  genuinely low-data or rapidly-changing-category setting, despite lower accuracy here.
- **DistilBERT and Word2Vec** both add real complexity and compute cost that this
  dataset's results don't justify, worth knowing before defaulting to "use BERT" as
  a first instinct on a well-separated classification task.

## Limitations

- All four methods were evaluated on a single, relatively clean, well-known dataset.
  Results may not generalize to noisier or more class-overlapping text.
- DistilBERT fine-tuning showed real overfitting risk on a dataset this small; only
  checkpointing and best-epoch selection avoided reporting a worse-than-optimal result.
- Zero-shot performance depends on prompt wording and the specific LLM used; results
  are not necessarily representative of zero-shot classification in general.
- One zero-shot response (of 334) was unparseable and defaulted to the majority class.

## Running this notebook

1. Open the [dataset on Kaggle](https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive)
   and create a new notebook from the dataset page.
2. Enable GPU (Settings → Accelerator) for the DistilBERT section, and Internet
   (Settings → Internet) for the Word2Vec download.
3. Add a `GROQ_API_KEY` secret (Add-ons → Secrets) for the zero-shot section, get a
   free key at [console.groq.com](https://console.groq.com).
4. Run top to bottom.

## Repo structure

```
bbc-news-classifier/
├── bbc_news_classification.ipynb   # full pipeline: data, four methods, evaluation
├── README.md
└── requirements.txt
```
