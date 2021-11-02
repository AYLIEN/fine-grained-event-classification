# Blog: Zero-Shot Event Classification for Newsfeeds

By Chris Hokamp and Demian Gholipour Ghalandari

#### TLDR
* we share a simple, effective and scalable approach for zero-shot event classification
* we give an overview of zero-shot learning for fine-grained event classification and the CASE 2021 shared task
* we provide jupyter notebooks with code and examples

#### What you can do with this work

* Start building a zero-shot classifier by writing down descriptions of events you’re interested in
* Apply our system to classify news with your custom labels

We hope readers come away with a clear understanding of how easy it is to create reasonably efficient zero-shot text-classification models that are good baselines for many real world tasks.

----------------- 
### The News as a Stream of Events

We intuitively think of news as a time-series of discrete events. 
For example, we might visualize yesterday’s top news events like this:

<p align="center">
  <img src="../diagrams/news-events.png" alt="drawing" width="300"/>
</p>

However, raw streams of news events, such as the RSS feeds of major news publishers, are very noisy. 
Humans are good at contextualizing information and understanding what is useful, but we aren't good at 
processing high volumes of content, and we don't scale well. 

<p align="center">
  <img src="../diagrams/manual-news-event-extraction.png" alt="drawing" width="600"/>
</p>

### Event Classification

We would like to build automatic ways to filter a raw stream of news into a feed which only contain events that are (1) relevant and 
(2) novel for a given user. This post focuses on (1): classifying news events as relevant / not relevant to specific topics. 

We will use machine learning models for text classification and let users both **define** and **subscribe** to labels of interest. 
This is similiar to following particular topics on sites such as Google News, with the important
distinction that we do not want to miss *any* events of a certain type. In other words, we are not building a recommender system, 
we are building a ML-driven event monitoring system for _**filtering**_ news content, and both precision and recall are important.

[//]: # (TODO: note that recommender systems intuitively go for precision and usually don't worry about recall)

#### Detecting Event Types

To build our news event monitoring system, we will need a way of classifying news events according to their type. 
We can approach this as a standard text-classification task, but with an interesting twist: we may not know the types of events up-front. 
In other words, we want to design a pipeline that supports the addition of new labels on-the-fly.

In another twist: we may not have _**any**_ training data at all for the classes we want to detect. We might just have 
a class label, and possibly a short snippet of text describing the label. 

And finally, our news stream is _**really**_ big: we're looking for solutions that scale to millions of articles per day, 
while easily handling hundreds or thousands of distinct labels, each of which may have a very different expected volume 
per day. 

Amazingly, it is actually possible to build a simple baseline system that satisfies these requirements.
It won't outperform usecase-specific systems that use expensive models and well-tuned hyperparametes along 
with substantial in-domain training data, but it will serve as a good baseline for any explorations in text classification, 
especially for usecases where scalable support for zero-shot classification is an essential requirement. And we can get it up 
and running in less than five minutes(!). 

### Key Ingredients

With the requirements set out in the previous section in mind, let's get specific. We're going to build a nearest-neighbor based 
zero-shot classifier. 

We'll need:
- a good vectorizer for snippets of news text and label descriptions
- a fast search index for looking up the most similar items to a query

### Nearest-Neighbors Based Zero-Shot Classification

Zero-shot classification settings are characterized by the lack of any labeled examples for the classes of interest. 
Instead, each class is usually represented through meta-information about the class itself, e.g. a short textual class 
description in the case of text classification. 
We are interested in this setup because it simplifies the baseline event classification workflow a lot: 
if a user comes to us with a new event type, we want to be able to immediately start serving them news events of 
that class without needing to collect a new labelled dataset or go through a complicated re-training/tuning stage. 

Since we’re sciency types, obviously we want to use cool machine learning models.
And since we’re engineers we want the model we use to be fast, cheap, and scalable. 
So in order to build our baseline system, we’re going to constrain ourselves to the simplest type of model, 
but we’re going to be clever about how we set things up.

The core idea of many zero-shot text classification methods is to **compare** a text snippet to a label description. 

In this particular task, label descriptions are for example "Man-made disaster" or "Peaceful protest". 
A powerful zero-shot model could be trained to jointly "read" both the text snippet and a 
label description to output a score indicating how closely related they are. 

In a nutshell, we wish to have a classifier that can solve this problem:
```
Input: snippets of text
Output: snippets labeled according to a taxonomy of event types
No training data
High Throughput
```

[//]: # (TODO: cite NLI-based models)
Some recently published zero-shot models use cross-encoders pre-trained on the NLI task. Although these models perform well,
they do not meet our scalability requirements, because a cross-encoder would require passing every possible combination of snippet + candidate-label. 
through the model. Instead, we will use a bi-encoder which embeds snippets and label descriptions in to the same embedding space. 

<p align="center">
  <img src="../diagrams/sentence-transformers-Bi_vs_Cross-Encoder.png" alt="drawing" width="600"/>
</p>
Image from [Sentence Transformers documentation](https://www.sbert.net/examples/applications/cross-encoder/README.html#when-to-use-cross-bi-encoders)


[//]: # (TODO: point reader to cross- vs bi- encoders)

However, doing pairwise comparisons between thousands or millions of text snippets and tens or 
hundreds of labels is computationally expensive.

Instead, we will encode text snippets and label descriptions separately into embeddings of the same vector space. Using these embeddings, we measure the cosine similarity between a snippet and each label. We classify a snippet by simply picking the label with the closest embedding. A crucial requirement is to have a powerful vector representation. We use a model from the [sentence-transformers](https://www.sbert.net/) library to vectorize text snippets and labels. We summarize our approach as follows:
1) Encode label descriptions, store label embeddings
2) Classify a new text snippet:
  - encode snippet
  - measure cosine between snippet embbedding and label embeddings
  - pick label with closest embedding

If we have hundreds or thousands of labels, measuring the cosine to every single label can also be become expensive - however, 
there is a simple fix: we can use *approximate nearest-neighbor search* to find the closest label(s). 

The approach naturally supports dynamic labels: we simply add or remove labels and their embeddings from our storage.

Here is a diagram of this framework:

<p align="center">
  <img src="../diagrams/zero-shot-baseline.png" alt="drawing" width="500"/></div>
<p>

In this post, we'll go over an approach to zero-shot event classification that worked well at 
the CASE 2021 fine grained event detection shared task. Code and examples are available
in [our project repository](https://github.com/AYLIEN/fine-grained-event-classification).  

**The shared task**
  
### Zero-shot Event Classification

To test our system, we'll focus on a specific text-classification task: zero-shot fine-grained event classification. 
We participated in Task 2 of the CASE 2021 shared task: Fine-Grained Event Classification.
Shared tasks are a great way to test and share ideas in a fair and open setting, 
and to get fast feedback about how different approaches stack up. Many thanks to the organizers of the CASE 2021 
shared task for all of the hard work they did.  

The CASE fine-grained event classification shared task is an ideal challenge for testing zero-shot text classification models. 
The task is to classify short text snippets that report socio-political events into fine-grained event types. 
These types are based on the Armed Conflict Location & Event Data Project (ACLED) event taxonomy, 
which contains 25 detailed event types, such as “Peaceful protest”, “Protest with intervention”, or “Diplomatic event”.
  
We submitted several systems to the CASE 2021 shared task to get an idea how our models stack up in an unbiased evaluation setting. 
The model described above worked best.

**Results**
  
The CASE shared task organizers picked 5 event types for zero-shot experiments. All submitted systems had to classify examples of these types without having seen training examples of these. Our best system produced the following average evaluation scores over these labels:

|          | Precision | Recall | F1-Score |
|----------|-----------|--------|-----------|
| **micro**    | 0.840     | 0.358  | 0.502     |
| **macro**    | 0.914     | 0.383  | 0.477     |
| **weighted** | 0.920     | 0.358  | 0.443     |

Based on the weighted F1-Score, our system was the best among several zero-shot approaches when we submitted it.

### Transformers vs. Word2Vec

One of our important takeaways from this work was that transformer-based embedding models really are a lot better than word2vec-based embedding. 
However, we are embedding short snippets of text in this task, so these results might not hold if we were processing whole documents. 
Also, transformer-based models are a lot more resource intensive, so there will likely always be some tradeoff between model 
performance and throughput in production settings. 

**Our Code**

#### Notebooks 
Check out our implementation in [this notebook](../notebooks/SentenceTransformers-ZeroShot-Baseline.ipynb) 
and use it to build a custom classifier.

This notebook sets up the dataset for the fine grained shared task, and implements a zero-shot prediction model, 
which uses the [paraphrase-multilingual-mpnet-base-v2](https://www.sbert.net/docs/pretrained_models.html) 
from the [sentence transformers library](https://www.sbert.net/index.html). This excellent library and repository of pre-trained models 
provides a great starting point for prototyping zero-shot text classification systems. 

We simply embed each of the labels using its meta-data and we are immediately ready to classify. 

There are additional notebooks available in the `notebooks/` directory that we plan to discuss in the second post of this series.

From a pedagogical perspective, we believe this approach may be even more intuitive for newcomers to deep learning and 
document embedding than the supervised view of K-nearest-neighbors models that is often the first topic that is introduced in applied ML courses.
In the design we have outlined above, we are effectively treating each label's description as a weakly-labeled training instance, 
and creating a KNN classifier with `K = 1` and exactly one candidate for each label in the output space. 
  
If you'd like to test this approach with other news data, have a look at some of the [Aylien topical datasets](https://aylien.com/resources/datasets).


### Conclusion

In practice, creating performant and scalable NLP models for real products usually requires iteration 
on both datasets and models, and any off-the-shelf solution will seldom hold up to the combination of domain knowledge,
data annotation, and real-world ML experience. 


## References

Case 2021 Task 2: Fine-grained Event Classification Github repo 
https://github.com/emerging-welfare/case-2021-shared-task/tree/main/task2

Jakub Piskorski, Jacek Haneczok, Guillaume Jacquet
New Benchmark Corpus and Models for Fine-grained Event Classification: To BERT or not to BERT?
https://aclanthology.org/2020.coling-main.584/

Yu Meng, Yunyi Zhang, Jiaxin Huang, Chenyan Xiong, Heng Ji, Chao Zhang, Jiawei Han
Text Classification Using Label Names Only: A Language Model Self-Training Approach.
https://aclanthology.org/2020.emnlp-main.724/

Fine-grained Event Classification in News-like Text Snippets-Shared Task 2, CASE 2021
J Haneczok, G Jacquet, J Piskorski… - Proceedings of the 4th …, 2021 - aclanthology.org
This paper describes the Shared Task on Fine-grained Event Classification in News-like
Text Snippets. The Shared Task is divided into three sub-tasks:(a) classification of text
snippets reporting socio-political events (25 classes) for which vast amount of training data …
https://aclanthology.org/2021.case-1.23/

CASE 2021 Task 2 Socio-political Fine-grained Event Classification using Fine-tuned RoBERTa Document Embeddings
S Kent, T Krumbiegel - Proceedings of the 4th Workshop on …, 2021 - aclanthology.org
We present our submission to Task 2 of the Socio-political and Crisis Events Detection
Shared Task at the CASE@ ACL-IJCNLP 2021 workshop. The task at hand aims at the fine-
grained classification of socio-political events. Our best model was a fine-tuned RoBERTa …
https://aclanthology.org/2021.case-1.26/



-----------
## Buffer

We can trade-off performance for speed as needed by using more efficient vectorizers.   

#### Why we care about event classification at Aylien

There are many reasons I might be monitoring a stream of news for a certain type of event -- 
I might be looking for events that would impact the supply chain of a particular business, or for events 
that are likely to impact political decisions in a certain region. 

One of the most common reasons to monitor the news at scale is to filter for new risks related to a 
particular business vertical such as Environmental-Social-Governance related risks. 
In each of these examples, I am looking for news events that meet a certain criteria. 
One way to specify the type(s) of event I'm looking for is to use an existing taxonomy of event types 
such as the one used by [ACLED](https://acleddata.com/). 

For each new piece of text, one of the first things we may ask is "does this contain a event?". In other words, does 
this piece of content describe an event. If it does, then we  "what type of event is this?". 
Defining what an `Event` is is notoriously challenging, but in this work we will stick to discrete occurences over short timespans, 
following the ACLED taxonomies. Implicitly or explicitly, we have a taxonomy of event types, 
and we would like to put this piece of content in its place, or ignore it if it isn't one of the things we're looking for.

But often, we are really only interested in certain _kinds_ of events. In other words, in my personal feed, 
I only want to see events meeting certain criteria. For example, I might only care about monitoring a conflict or 
geopolitical crisis in a certain region. I want to filter the news stream, removing everything that doesn't 
meet the criteria of an interesting event. There are many ways we might approach this, 
but one of the most straightforward is to label each piece of content with one or more 
labels indicting what type of event happened.

We call the task of labeling each event in a stream with its type **event classification**.

In practice, news is a stream of data, but on a more useful level of abstraction, news is a stream of events. A timeline with discrete events ordered by time is a  good mental model for what the news is. 

In news intelligence, we may also make a distinction between tagging based on content or categories, and instantiating specific events such as terrorist attacks or geopolitical conflict. 


Visualize: tags vs event schemas
Visualize: a timeline of terrorist attacks 

Note: event duration, etc are out-of-scope


Note the distinction between what happened (events) vs what the content is about (topics). 
This is important because I don't want to see content that is a general discussion of geopolitical conflict, I only 
want my stream to capture specific instances of conflict. I want to get pinged when a new terrorist attack happens, but I don't 
want to get pinged about general discussions or opinion pieces related to terrorism. Let's try to make our mental model explicit -- 
I have an idea of what an event is/isn't. Different types of events have different properties in their schemas, 
and instantiations of the schema will be discrete events.

Connect to pattern matching in FP and OOP
Frame Semantics and FP/OOP
