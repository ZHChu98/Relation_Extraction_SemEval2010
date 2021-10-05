# **Relation Extraction On SemEval 2010 Task 8**

## **Brief Description**

Relation Extraction (RE) is a vital task in Natural Language Processing (NLP). In this task, it aims to distinguish the relation type between two nominals given in the text and this technology has been widely applied in a large variety of fields, such as constructing Knowledge Base System, Biomedical Data Mining, Question Answering, etc. Our main goal is to develop a stand-out supervised deep learning approach to realize RE on the SemEval 2010 Task 8 dataset.

## **Project Structure**

1. Proprocess the data using *CoreNLP* and *SpaCy*, such as entity labeling (BILOU tags), dependency parsing, pos-tagging
2. Create the joint model which consists of two subnetwork: a left-to-right entity detection network, and a tree-based biLSTM relation classification network
3. Feed the joint model with the data and tune the model by dropout, scheduled learning rate, L2 regularization, and negative sampling
4. Compare the performance with previous deep learning approaches

for more information, please view the [report](Report.pdf).
