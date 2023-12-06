# toxic_comment_group2
# Presentation 2 - Group 2
## Can we use pre-trained models to detect toxic comments?

### Introduction
Hugging Face is a community for learning about and advancing machine learning. One can find many machine learning models, datasets and demo apps on Hugging Face Hub. It’s purpose is to help the community to advance Open, Collaborative, and Responsible Machine Learning. (Hugging Face, n.d.) 
“A pre-trained model is a saved network that was previously trained on a large dataset […] You either use the pretrained model as is or use transfer learning to customize this model to a given task” (TensorFlow, n.d.).
Using pretrained models is very Convenient and Time-saving. There is no need to build a model from scratch, thus making the data processing and analyzing much faster (Baeldung, March 2023).

### Data
The data for this model training were retrieved from kaggle.com. It is a manually labeled toxicity data collection with a thousand comments that was retrieved from YouTube videos concerning the 2014 Ferguson disturbance by Reihaneh Namdari. It has a 10.00 usability with 12.1K views. 

### Model
we are using a text classification model called martin-ha/toxic-comment-model (https://huggingface.co/martin-ha/toxic-comment-model) from huggingface.com. It has more than 1,000,000 downloads over the last month with 94% accuracy when tested with Jigsaw Unintended Bias in Toxicity Classification (https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data). It is a fine-tuned version of the DistilBERT model to classify toxic comments. 

### Conclusion


### Python Libraries
pandas
transformers
datasets
torch
matplotlib
tqdm
NumPy
emoji
re
sklearn.model_selection
seaborn

### Contributors
Nguyen Mai Linh (Sandy) 20229031
Farhan Toshi Hermawan 20229531
Fariha Qorinatuz Zahra 21229529
