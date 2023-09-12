# Basic concepts

Here are some high level concepts to give you an idea on how Crank can be used. 

## Deep Learning First

The focus of Crank is to provide tools to enable tabular machine learning via mini-batch learning. In general, we presume we are learning over a large dataset, too large to fit in memory and will leverage deep learning techniques to ensure it is appropriate. 

The data processed presumes everything occurs in an **online** fashion, rather than the typical traditional approach where it is available as a whole batch and processed altogether at one time.

## Single File Per Algorithm

This repository is for pedagogy purposes, but may be useful for you! Each implementation is designed to be part of its own standalone file, and aims to fulfil _vanilla_ pytorch as much as possible. This is to ensure there is minimal dependencies and maximum flexibility to alter and use these algorithms to suit user needs. Infact, we expect users to simple take individual files and adapt the models as required. 

## Input Data Types

As we're reliant on vanilla pytorch, the training data is purely numeric. This may be overly restrictive in the real-world, though hopefully in conjunction with typical preprocessing pipelines it will be possible to have a _reasonable_ experience when combined with libraries like [lightning](https://lightning.ai/).