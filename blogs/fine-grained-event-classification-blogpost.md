# Blog: Zero-Shot Event Classification in News

In practice, news is a stream of data, but on a more useful level of abstraction, news is a stream of events. A timeline with discrete events ordered by time is a  good mental model for what the news is. Given a new piece of news content, one of the first things we may ask is "does this contain a new event?", then, if it does, "what type of event is this?". Defining what an `Event` is/isn't is notoriously challenging, but in this work we will stick to discrete occurences over short timespans, following the ACLED taxonomies. Implicitly or explicitly, we have a taxonomy of event types, and we would like to put this piece of content in its place.

In News Intelligence, we may also make a distinction between tagging based on content or categories, and instantiating specific events such as terrorist attacks or geopolitical conflict. 

Visualize: tags vs event schemas
Visualize: a timeline of terrorist attacks 

Note: event duration, etc are out-of-scope

In this post, we'll go over an approach to zero-shot event classification that worked well at the CASE 2021 fine grained event detection shared task. 


#### Why we care about event classification

We often think of news as a time-series of discrete events. For example, we might visualize yesterday’s top news events like this:

<img src="../diagrams/news-events.png" alt="drawing" width="275"/>

But often, we are really only interested in certain _kinds_ of events. In other words, in my feed, I only want to see events meeting certain criteria. For example, I might only care about monitoring a conflict or geopolitical crisis in a certain region. I want to filter the news stream, removing everything that doesn't meet the criteria of an interesting event. There are many ways we might approach this, but one of the most straightforward is to label each piece of content with one or more labels indicting what type of event happened.

Note the distinction between what happened (events) vs what the content is about (topics). This is important because I don't want to see content that is a general discussion of geopolitical conflict, I only want my stream to capture specific instances of conflict. 

Note: the importance of zero-shot for this

We call the task of labeling each event in a stream with its type **event classification**.

#### What we did about it

Since we’re sciency types, of course we want to use a machine learning model to do this. And since we’re engineers we want the model we use to be fast, cheap, and scalable. So we’re going to constrain ourselves to the simplest type of model, but we’re going to be clever about how we set things up.

**The task**

Luckily there’s a shared task for that, the CASE fine-grained event classification shared task. The task is to classify short text snippets that report socio-political events into fine-grained event types. These types are based on the Armed Conflict Location & Event Data Project (ACLED) event taxonomy, which contains 25 detailed event types, such as “Peaceful protest”, “Protest with intervention”, or “Diplomatic event”.

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

**Our Code**

Check out our implementation in [this notebook](../notebooks/SentenceTransformers-ZeroShot-Baseline.ipynb) and use it to build a custom classifier.

#### What you can do with our work:

* Start building a zero-shot classifier by writing down descriptions of events you’re interested in
* Apply our system to classify news with your custom labels
