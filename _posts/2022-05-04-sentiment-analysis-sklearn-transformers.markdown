---
layout: post
title:  "Sentiment Analysis for Movie/TV Reviews using classical and deep learning"
subtitle: "Comparing a classical machine learning approach and a deep learning transformer model approach for categorizing movie/tv reviews into positive, negative, or irrelevant classes"
date:   2022-05-04
category: nlp
tags: transformers sklearn huggingface class-competition sentiment-analysis
---

[*Crossposted from Medium*](https://medium.com/@jayyydyyy/sentiment-analysis-for-movie-reviews-using-classical-and-deep-learning-9aa1183c9825)

![Approximately what we're trying to analyze sort of… 😋](/assets/img/fakemoviereview.png)

As we are people with innate linguistic intuitions, the task of dividing something like tv/movie reviews into positive or negative reviews may seem easy, maybe even trivial.
Given its trivial nature, imagine being *assigned* to classify millions of movie reviews. 
Now, a once singularly easy task has become *unmanageable and unruly*. 

Computers do not possess this same innate linguistic intuition, so how can we leverage modern advancements in artificial intelligence, 
specifically natural language processing, to do the heavy lifting of this task for us?

## Task

Continue to imagine this scenario where you are assigned to classify tv/movie reviews ad nauseum. 
This task requires you categorize those reviews into *three categories*:

- **Not a TV/Movie Review** (represented by a 0)
- **A Positive TV/Movie Review** (represented by a 1)
- **A Negative TV/Movie Review** (represented by a 2)

Now, we're going to be training the machine learning models to do this task for us, so we need a dataset that represents some of this work already done for us.
Specifically, we have a csv file (***train.csv***) containing 3 columns:

- **ID** : A unique ID associated with movie review
- **TEXT** : The actual text of the review.
- **LABEL** : The label for the category (0,1,2 corresponding with the categories mentioned above)

Additionally, we have another csv file (***test.csv***) containing only 2 columns, the ID and TEXT. 
This conceptually represents the task of having to categorize the reviews ourselves. 
After training our models, we will predict the categories associated with those reviews and submit that to whomever assigned us this task. 

This data is a modified version of this dataset:

```
@inproceedings{pang-lee-2004-sentimental,
    title = "A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts",
    author = "Pang, Bo and Lee, Lillian",
    booktitle = "Proceedings of the 42nd Annual Meeting of the Association for Computational Linguistics ({ACL}-04)",
    month = jul,
    year = "2004",
    address = "Barcelona, Spain",
    url = "https://aclanthology.org/P04-1035",
    doi = "10.3115/1218955.1218990",
    pages = "271--278",
}
```

To accomplish this task, I'll be comparing a few methods and then opting for the approach that yields the greatest performance.


## Exploratory Data Analysis

To begin, we need a way to work with this data. 

We can use the python library *pandas* to read the ***train.csv*** into a useable format called a *DataFrame*.

```python
train_path = 'data/train.csv'

df = pd.read_csv(train_path)
```

Then we can begin some exploratory analysis.

Let's start by observing some key properties:

```python
print("Total rows:", df.shape[0],'\n')

print("Empty values in column ID:",df['ID'].isna().sum())
print("Empty values in column TEXT:",df['TEXT'].isna().sum())
print("Empty values in column LABEL:",df['LABEL'].isna().sum(),'\n')

irrelevant_slice = df['TEXT'].loc[df['LABEL']==0]
pos_slice = df['TEXT'].loc[df['LABEL']==1]
neg_slice = df['TEXT'].loc[df['LABEL']==2]

print("Number of Irrelevant Reviews", len(irrelevant_slice))
print("Number of Positive Reviews", len(pos_slice))
print("Number of Negative Reviews", len(neg_slice),'\n')

print("Avg Length of Irrelevant Reviews", sum([len(text) for text in irrelevant_slice.dropna()]) / len(irrelevant_slice))
print("Avg Length of Positive Reviews", sum([len(text) for text in pos_slice.dropna()]) / len(pos_slice))
print("Avg Length of Positive Reviews", sum([len(text) for text in neg_slice.dropna()]) / len(neg_slice))
```

Output:
```
Total rows: 70187 

Empty values in column ID: 0
Empty values in column TEXT: 7
Empty values in column LABEL: 0 

Number of Irrelevant Reviews 35000
Number of Positive Reviews 17645
Number of Negative Reviews 17542 

Avg Length of Irrelevant Reviews 426.21105714285716
Avg Length of Positive Reviews 1332.3479739302918
Avg Length of Positive Reviews 1298.990822027135
```

From this it appears that we have a roughly even split of tv/movie reviews and irrelevant reviews. Among tv/movie reviews we have another even split between positive and negative reviews.

We have seven empty values in the TEXT column. Thats a pretty small portion, but just to be safe, let's fill them with empty strings that we can still pass into our models.

```python
df = df.fillna("")
```

We should also take a peak at some random samples from each class.
I'll highlight a few here:

### The Irrelevant:
>*This was a pretty good book.  I only wish i did not have to buy 2 more to finish it.....*

Irrelevant reviews seem to frequently mention other products apart from movies/tv. In this case, this review mentioned books.


### The Positive:

>*No music. No stupid masala. A reasonably realistic portrayal of the police system in India and based on a real "encounter" specialist in India, Daya Nayak. That is Ab Tak 56 (56 symbolises how many criminals the lead "Sadhu Agashe" has killed" - well you already know that bit)Brilliance exudes Nan Patekar in the role as a relaxed and calculating Indian cop. THe one liners are just hilarious. The plot though slightly predictable on review, is intriguing all the same. Another one of the films from Ram Gopal Vermas The Factory. Movies which are either decent or really good, Ab Tak CHappan meanders close to very good. But yet remains one of the Top 70 films released from India, commercial and artsy included.<br /><br />What is great is the story telling is relaxed and showcases finally (in an Indian flick) how the police network works. The cast is really damn good but seriously the one liners are funny as hell (though i dont know if the subtitled version will appear as funny) The producers are trying for a Cannes release, which is interesting. Made by debut director Shamit Aman (i think thats his name).<br /><br /> ... continued ...*

Immediately the length discrepancy we observed earlier is clear (and I've specifically chosen a shorter sample to share and crop my selections too). 

In the positive we see many positive sentiment adjectives and adverbs: *reasonably*, *brilliance*, *hilarious*, *good*, etc..

### The Negative:
>*I gave this a 3 out of a possible 10 stars.<br /><br />Unless you like wasting your time watching an anorexic actress, in this film it's Carly Pope, behaving like a ditz, don't bother.<br /><br />Carly Pope plays Sara Novak, a young college student, who becomes intrigued with a game of riddles, that leads her down into subway tunnels underneath the city - a dangerous thing for even a well-armed man to go in alone.<br /><br />There are various intrigues in the film <br /><br /> ... continued ...*

Similar to the positive, we see a much longer review than non-movie reviews. 

Here we see negative sentiment phrases like *wasting your time* and probably most importantly the imperative *don't bother*.

---

Additionally, both positive and negative samples contain *ratings* and therefore I posit that higher numbers like *7* could lean towards classifying something as positive, while the presence of a number like *3* could lean towards classifying as negative. This of course depends on our tokenization strategy and whether or not we even include single digit numbers.

Overall, it seems it will be much easier to distinguish between irrelevant and relevant reviews than it is for positive and negative. This is due to the major differences and length, format, and content between irrelevant and relevant reviews. 
However, between positive and negative, the qualities of length, format, and content are all similar. We will instead have to rely on finding these characteristic positive and negative words or phrases.



## scikit-learn Approach
To begin we need to separate our training data into two parts, the training data and the *development data*. 
This will allow us to evaluate our models by checking our predictions against the actual reported values in the development set.

```python
# Here I opt for a 85-15 split between train and development sets
train_df, dev_df = train_test_split(df,test_size=0.15,random_state=42)
```

Next we can fit the text to a scikit-learn TfidfVectorizer and get a sparse matrix of TF-IDF values for each word for each document.

```python
featurizer = TfidfVectorizer(ngram_range=(1,1),use_idf=True,sublinear_tf=True)
featurizer.fit(train_df['TEXT'])

features = featurizer.transform(train_df['TEXT'])
```

For this problem I'm going to try three of the popular classical machine learning models included with scikit-learn's machine learning libraries:

- Logistic Regression
- Multinomial Naive Bayes
- Linear Support Vector Machine

I'll setup another class to handle training and fitting all of these models:

```python
class Classifier:
    def __init__(self) -> None:
        self.lr_model = LogisticRegression(max_iter=500)
        self.nb_model = MultinomialNB()
        self.svm_model = LinearSVC(max_iter=2500)
    def fit(self, features, labels):
        self.lr_model.fit(features,labels)
        self.nb_model.fit(features,labels)
        self.svm_model.fit(features,labels)

    def predict(self, features):
        return self.lr_model.predict(features),self.nb_model.predict(features),self.svm_model.predict(features)

clf = Classifier()

clf.fit(features,train_df['LABEL'])

lr_preds,nb_preds,svm_preds = clf.predict(featurizer.transform(dev_df['TEXT']))
```

## scikit-learn Evaluation

Once we have our predictions, we can evaluate them for each model on the common metrics of precision, recall, and f1-score.

```
Linear Regression : -------------------
              precision    recall  f1-score   support

   Not Movie       0.98      0.99      0.98      5247
    Positive       0.88      0.87      0.88      2710
    Negative       0.89      0.88      0.89      2572

    accuracy                           0.93     10529
   macro avg       0.92      0.91      0.92     10529
weighted avg       0.93      0.93      0.93     10529

Multinomal NB : -------------------
              precision    recall  f1-score   support

   Not Movie       0.96      0.98      0.97      5247
    Positive       0.86      0.78      0.82      2710
    Negative       0.83      0.88      0.85      2572

    accuracy                           0.90     10529
   macro avg       0.88      0.88      0.88     10529
weighted avg       0.90      0.90      0.90     10529

Linear Support Vector Machine : -------------------
              precision    recall  f1-score   support

   Not Movie       0.98      0.99      0.99      5247
    Positive       0.88      0.88      0.88      2710
    Negative       0.89      0.88      0.89      2572

    accuracy                           0.93     10529
   macro avg       0.92      0.92      0.92     10529
weighted avg       0.93      0.93      0.93     10529
```

After running the metrics evaluation, we can see that the ***Support Vector Machine*** model wins out just by a bit over the Logistic Regression model.


## Huggingface 🤗 Transformers Approach

Now that we've tried the classical machine learning approach using scikit-learn, 
it's time to indulge in the current state-of-the-art for natural language processing: *transformer models*.
Using Huggingface🤗's transformers library, we can achieve some pretty incredible results with minimal effort!
For this approach, I'm going to import and fine-tune a distilled version of BERT available on Huggingface's hub, *DistilBERT*.

Specifically, I'll be using *distilbert-base-uncased*.
Please refer to the huggingface hubs page for distilbert-base-uncased for more details on the model including its intended uses, problems with bias, and evaluation on the raw model.

https://huggingface.co/distilbert-base-uncased

---

We should start by getting our data in order. We can reuse those pandas dataframes that we split up earlier and cast them as huggingface Datasets.

Although noteably, trying to run empty strings through the transformer model will cause some errors, so let's change our strategy to add some filler text. 

```python
from sklearn.model_selection import train_test_split
from datasets import Dataset

df = pd.read_csv(train_path)
df = df.fillna("NONE")

# Here I opt for a 85-15 split between train and development sets
train_df, dev_df = train_test_split(df,test_size=0.15,random_state=42)

train_data = Dataset.from_pandas(train_df)
dev_data = Dataset.from_pandas(dev_df)
```

We're also going to need some preprocessing steps to get everything ready for the model:

```python
from transformers import AutoTokenizer, DataCollatorWithPadding

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# The case for the TEXT column doesn't matter all that much, but distilbert is expecting a lowercase 'label' column
train_data = train_data.rename_columns({'TEXT':'text','LABEL':'label'})
dev_data = dev_data.rename_columns({'TEXT':'text','LABEL':'label'})

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

train_data = train_data.map(preprocess_function, batched=True)
dev_data = dev_data.map(preprocess_function, batched=True)
```

So here we have imported the tokenizer associated with the pretrained distilbert model.
We've also setup a data collator to handle padding. 

We can then use the preprocess function and Dataset.map to tokenize our Datasets.

Now that we've got our data in order, we just need to import our model to train.

```python
from transformers import AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
model.to('cuda:0')
```

This will download the pretrained distilbert model and set it to expect 3 labels for further training!
We also send the model to the first cuda device. For simplicity I've trained this model in google colaboratory and have my runtime set to GPU to increase performance.

Now finally we can fine tune our model on our training data!

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="./results",learning_rate=2e-5,
                                  per_device_train_batch_size=8, per_device_eval_batch_size=8, 
                                  num_train_epochs=3, weight_decay=0.01)
trainer = Trainer(model = model, args = training_args, train_dataset=train_data, eval_dataset=dev_data,data_collator=data_collator)

trainer.train()
```

This is also really easy using the pyTorch version of Huggingface🤗 transformers. All we need to do is initialize Trainer and TrainerArguments objects with our desired hyper-parameters and then let it run.

Now, I'm impatient (and worried that google colab will disconnect on me) so I only ran my trainer for less than a full epoch clocking in at 1 hour and 15 minutes.

But now we can save our model's current checkpoint:

```python
# I'm saving mine to my mounted google drive in colab
trainer.save_model('/content/drive/MyDrive/Ling539/submission_checkpoint')
```

## Transformer Evaluation

And that's it for training! Now we just need to evaluate the model and compare it to our scikit-learn approach.

```python
import numpy as np
y = trainer.predict(dev_data)
preds = np.argmax(y.predictions,axis=1)

# A helper function I wrote
metric_printout("Distilbert Finetuned Transformer Evaluation", preds, dev_data['label'])
```

```
Distilbert Finetuned Transformer Evaluation : -------------------
              precision    recall  f1-score   support

   Not Movie       1.00      0.99      0.99      5267
    Positive       0.93      0.91      0.92      2768
    Negative       0.90      0.93      0.92      2494

    accuracy                           0.95     10529
   macro avg       0.94      0.94      0.94     10529
weighted avg       0.95      0.95      0.95     10529
```

Thats definitely a bit better than the scikit-learn approach- we managed to get at or above 0.90 for each metric!

## Conclusion

Our finetuned *DistilBERT* model definitely outperforms our scikit-learn classical machine learning models.
However, for this task if you compare the transformers model to specifically the support vector machine model, we really only see a marginal difference. 

There is *information* we were unable to capture and represent in both models! However the way we approach understanding that information for both approaches would be much different.

For the linear support vector machine, we could most likely see vast improvements with some clever feature engineering. 
Currently we're just feeding it an incredibly sparse matrix of TFIDF scores for each document. 
Some more effort could be put into reducing the dimensionality of that matrix by excluding uninformative words and maybe performing a dimensionality reduction technique like [latent semantic analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis) on it. 
Additionally, we could potentially provide more information by supplementing our matrix with features created from word vectors, thus encapsulating more semantic information.

For the transformer approach, with more time and resources we could have kept training to reach a lower loss. I believe we reached a local minimum as seen here:
```
Step	Training Loss
500	    0.333200
1000	0.213200
1500	0.199800
2000	0.193000
2500	0.192600
3000	0.189900
3500	0.187600
4000	0.193100
4500	0.189700
5000	0.173800
5500	0.178900
6000	0.181100
6500	***0.169900***
7000	0.176900
```

At step 6500 we hit a significant minimum and then curve back up again. 
However, I'm pretty sure we could reach an even lower loss given some more training time, as that is common with more robust neural net models like transformer models. 

This concept is known as ***double descent*** and is certainly outside the scope of this blog post, but you can [read here for more information](https://mlu-explain.github.io/double-descent/).


## Source Code and Reproducibility

[You can find my code here!](https://github.com/jayyydyyy/ling-539-sentiment-analysis-competition)

If you'd like to reproduce my results, make sure to install all the dependencies listed in the requirements.txt
However, for the transformers notebooks, I would recommend running them in google colaboratory or some other remote environment with GPU acceleration. 
You can always install and setup CUDA support on your local machine if you have a supported graphics card, but not everyone does! :)

For these reasons, I've included a jupyter notebook and a link to a google colab notebook for each approach.