# Blog: Zero-Shot Event Classification in News


#### TLDR -- What is this post about?
* Zero-shot learning, fine-grained event classification and the CASE 2021 shared task
* A simple, effective and scalable approach for zero-shot event classification
* Notebooks with code and examples

We hope readers of this post come away with a clear understanding of how easy it is to create a 
reasonably efficient zero-shot text-classification model that is a good baseline for many tasks.

--- 

### A Stream of Events

We often think of the news as a time-series of discrete events. For example, we might visualize yesterday’s top news events like this:

<img src="../diagrams/news-events.png" alt="drawing" width="275"/>

<img src="../diagrams/manual-news-event-extraction.png" alt="drawing" width="275"/>

However, the raw stream of news events is very noisy. Humans are good at contextualizing information and understanding what is useful, but we aren't good at processing lots of content, and we don't scale well. So we need automatic ways to filter the raw stream of events to only contain news that is relevant to us.  One way of filtering is to use machine learning models for text classification, and to only subscribe to certain labels that are assigned by our models.

To build our news event monitoring system, we will need a way of classifying events according to their type. We can approach it as a text-classification task, with an interesting twist: we may not know the types of events upfront. In other words, we want to design a pipeline that supports the addition of new labels on-the-fly.

In addition, our news stream is _really_ big: we're looking for solutions that scale to millions of articles per day, while considering hundreds or thousands of labels. 

In this post, we're focusing on a specific text-classification task: zero-shot fine-grained event classification. We're going to discuss our participation in the CASE 2021 Fine-Grained Event Classification shared task. Shared tasks are a great way to test ideas in a fair and open setting, and to get fast feedback about how different approaches stack up. 

### The Zero-shot Event Classification Task

#### The Setting
Zero-shot classification setups are usually defined by the lack of examples labelled with classes of interest. 
Instead, they usually represent each class through meta information, e.g. a short textual class description in 
the case of text classification. 
We are interested in this setup because it simplifies the event classification workflow a lot: 
if a user comes to us with a new event type, we want to be able to immediately start serving them news events of 
that class without needing to collect a new labelled dataset or go through a complicated re-training/tuning stage. 

In a nutshell, we wish to have a classifier that can solve this problem:
```
Input: snippets of text
Output: snippets labeled according to a taxonomy of event types
No training data
```

In this post, we'll go over an approach to zero-shot event classification that worked well at 
the CASE 2021 fine grained event detection shared task. 

#### Why we care about event classification

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

#### What we did about it

Since we’re sciency types, of course we want to use a machine learning model to do this. 
And since we’re engineers we want the model we use to be fast, cheap, and scalable. 
So we’re going to constrain ourselves to the simplest type of model, but we’re going to be clever about how we set things up.

**The shared task**

The CASE fine-grained event classification shared task is an ideal challenge for testing zero-shot text classification models. 
The task is to classify short text snippets that report socio-political events into fine-grained event types. 
These types are based on the Armed Conflict Location & Event Data Project (ACLED) event taxonomy, 
which contains 25 detailed event types, such as “Peaceful protest”, “Protest with intervention”, or “Diplomatic event”.

**Zero-shot classification**

We submitted several systems to the CASE shared task to get an idea how our models stack up in an unbiased evaluation setting. After that we did some more experiments to explore how to make our models even better, while still maintaining the efficient classification-via-similarity framework.

<div style="text-align:center"><img src="../diagrams/zero-shot-baseline.png" alt="drawing" width="500"/></div>

**What the results were**

The CASE shared task organizers picked 5 event types for zero-shot experiments. All submitted systems had to classify examples of these types without having seen training examples of these. Our best system produced the following average evaluation scores over these labels:

|          | Precision | Recall | F1-Score |
|----------|-----------|--------|-----------|
| **micro**    | 0.840     | 0.358  | 0.502     |
| **macro**    | 0.914     | 0.383  | 0.477     |
| **weighted** | 0.920     | 0.358  | 0.443     |

Based on the weighted F1-Score, our system was the best among several zero-shot approaches when we submitted it.

### Transformers vs. Word2Vec

One of our important takeaways from this work was that transformer-based embedding models really are a lot better than word2vec-based embedding. However, we are embedding short snippets of text in this task, so these results might not hold if we were processing whole documents. Also, transformer-based models are a lot more resource intensive, so there will likely always be some tradeoff between model performance and throughput in production settings. 


**Our Code**

#### Notebooks 
Check out our implementation in [this notebook](../notebooks/SentenceTransformers-ZeroShot-Baseline.ipynb) and use it to build a custom classifier.

This notebook works through setup and


There are additional notebooks available in the `notebooks/` directory that we plan to discuss in the second post of this series.

#### What you can do with our work:

* Start building a zero-shot classifier by writing down descriptions of events you’re interested in
* Apply our system to classify news with your custom labels

-----------
Buffer
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
