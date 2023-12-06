#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import gc
import os
import copy
import itertools
import re
import time
from random import choice, choices
from functools import reduce
from tqdm import tqdm
from itertools import cycle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from collections import Counter
from functools import reduce
from itertools import cycle
from scipy import stats
from scipy.stats import skew, kurtosis
from sklearn import metrics, model_selection, preprocessing, linear_model, ensemble, decomposition, tree
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import mean_squared_error, r2_score


# In[4]:


get_ipython().system('pip install tensorflow')


# In[ ]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[ ]:


INPUT_DIR = '/data'
train_logs = pd.read_csv(f'{INPUT_DIR}/train_logs.csv')
train_scores = pd.read_csv(f'{INPUT_DIR}/train_scores.csv')
test_logs = pd.read_csv(f'{INPUT_DIR}/test_logs.csv')
ss_df = pd.read_csv(f'{INPUT_DIR}/sample_submission.csv')


# In[ ]:


print(train_logs.shape)
print(train_logs['id'].nunique())
train_logs.head()


# In[ ]:


train_logs['wait_time'] = train_logs['down_time'] - train_logs.groupby('id')['up_time'].shift()
train_logs['wait_time'].fillna(0, inplace=True)
train_logs.head()


# In[ ]:


# Sum positive values, treat negative values as 0, and find counts
wait_time_df = train_logs.groupby('id')['wait_time'].agg(
    Total_Positive_wait_time=lambda x: x[x > 0].sum(),
    Total_Negative_wait_time=lambda x: x[x < 0].sum(),
    Count_Positive_wait_time=lambda x: (x > 0).sum(),
    Count_Negative_wait_time=lambda x: (x < 0).sum()
).reset_index()

wait_time_df.head()


# In[ ]:


action_time_df = train_logs.groupby('id')['action_time'].agg(
    Total_action_time=lambda x: x[x > 0].sum()
    ).reset_index()

action_time_df.head()


# In[ ]:


print(train_scores.shape)
print(sorted(train_scores['score'].unique()))
train_scores.head()


# In[ ]:


print(test_logs.shape)
print(sorted(test_logs['id'].unique()))
test_logs.head()


# In[ ]:


ss_df.head()


# In[ ]:


print(train_logs['activity'].unique())
print(train_logs['down_event'].unique())
print(train_logs['up_event'].unique())
print(train_logs['text_change'].unique())
print(train_logs['cursor_position'].unique())
print(train_logs['word_count'].unique())


# This notebook make use of essay reconstructor from https://www.kaggle.com/code/yuriao/fast-essay-constructor/notebook .

# In[ ]:


class EssayConstructor:
    
    def processingInputs(self,currTextInput):
        # Where the essay content will be stored
        essayText = ""
        # Produces the essay
        for Input in currTextInput.values:
            # Input[0] = activity
            # Input[1] = cursor_position
            # Input[2] = text_change
            # Input[3] = id
            # If activity = Replace
            if Input[0] == 'Replace':
                # splits text_change at ' => '
                replaceTxt = Input[2].split(' => ')
                # DONT TOUCH
                essayText = essayText[:Input[1] - len(replaceTxt[1])] + replaceTxt[1] + essayText[Input[1] - len(replaceTxt[1]) + len(replaceTxt[0]):]
                continue

            # If activity = Paste    
            if Input[0] == 'Paste':
                # DONT TOUCH
                essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
                continue

            # If activity = Remove/Cut
            if Input[0] == 'Remove/Cut':
                # DONT TOUCH
                essayText = essayText[:Input[1]] + essayText[Input[1] + len(Input[2]):]
                continue

            # If activity = Move...
            if "M" in Input[0]:
                # Gets rid of the "Move from to" text
                croppedTxt = Input[0][10:]              
                # Splits cropped text by ' To '
                splitTxt = croppedTxt.split(' To ')              
                # Splits split text again by ', ' for each item
                valueArr = [item.split(', ') for item in splitTxt]              
                # Move from [2, 4] To [5, 7] = (2, 4, 5, 7)
                moveData = (int(valueArr[0][0][1:]), int(valueArr[0][1][:-1]), int(valueArr[1][0][1:]), int(valueArr[1][1][:-1]))
                # Skip if someone manages to activiate this by moving to same place
                if moveData[0] != moveData[2]:
                    # Check if they move text forward in essay (they are different)
                    if moveData[0] < moveData[2]:
                        # DONT TOUCH
                        essayText = essayText[:moveData[0]] + essayText[moveData[1]:moveData[3]] + essayText[moveData[0]:moveData[1]] + essayText[moveData[3]:]
                    else:
                        # DONT TOUCH
                        essayText = essayText[:moveData[2]] + essayText[moveData[0]:moveData[1]] + essayText[moveData[2]:moveData[0]] + essayText[moveData[1]:]
                continue                
                
            # If activity = input
            # DONT TOUCH
            essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
        return essayText
            
            
    def getEssays(self,df):
        # Copy required columns
        textInputDf = copy.deepcopy(df[['id', 'activity', 'cursor_position', 'text_change']])
        # Get rid of text inputs that make no change
        textInputDf = textInputDf[textInputDf.activity != 'Nonproduction']     
        # construct essay, fast 
        tqdm.pandas()
        essay=textInputDf.groupby('id')[['activity','cursor_position', 'text_change']].progress_apply(lambda x: self.processingInputs(x))      
        # to dataframe
        essayFrame=essay.to_frame().reset_index()
        essayFrame.columns=['id','essay']
        # Returns the essay series
        return essayFrame


# In[ ]:


essayConstructor=EssayConstructor()
essay=essayConstructor.getEssays(train_logs)


# In[ ]:


print(essay.shape)
essay.head()


# In[ ]:


train_logs[train_logs['id'] == '0022f953'].head()


# In[ ]:


# Function to count words in a text
def count_words(text):
    return len(text.split())

essay['word_count'] = essay['essay'].apply(lambda x: count_words(x))
essay.head()


# In[ ]:


train_essay_score =  pd.merge(essay, train_scores, on='id')
train_essay_score.head()


# In[ ]:


def update_action(value):
    if value.startswith('Move From'):
        return 'All Move From'
    else:
        return value

# Create a new column with updated values
train_logs['Updated_activity'] = train_logs['activity'].apply(update_action)


# In[ ]:


action_counts = train_logs.groupby('id')['Updated_activity'].value_counts().unstack(fill_value=0)
action_counts.head()


# In[ ]:


train_essay_score = pd.merge(train_essay_score, action_counts, on='id', how='left')
train_essay_score.head()


# In[ ]:


train_essay_score = pd.merge(train_essay_score, wait_time_df, on='id', how='left')
train_essay_score.head()


# In[ ]:


train_essay_score = pd.merge(train_essay_score, action_time_df, on='id', how='left')
train_essay_score.head()


# In[ ]:


import nltk

nltk.download('punkt')

def count_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return len(sentences)

def count_paragraphs(text):
    paragraphs = [p for p in text.split('\n\n') if p.strip()]  # Split paragraphs and remove empty ones
    return len(paragraphs)


# In[ ]:


train_essay_score['Num_Sentences'] = train_essay_score['essay'].apply(count_sentences)
train_essay_score['Num_Paragraphs'] = train_essay_score['essay'].apply(count_paragraphs)
train_essay_score.head()


# In[ ]:


# Calculate average word count for each score
avg_word_count = train_essay_score.groupby('score')['word_count'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='word_count', y='score', data=avg_word_count, s=150, edgecolor='k', linewidth=1.5)

plt.title('Scatter Plot of Average Word Count for Each Score')
plt.xlabel('Average Word Count')
plt.ylabel('Score')

plt.savefig('Scatter_Plot_of_Average_Word_Count_for_Each_Score.png', dpi=300)

plt.show()


# In[ ]:


sns.set(style="whitegrid")

plt.figure(figsize=(14, 7))

# Loop through unique scores and create a histogram for each
for idx in range(0, len(train_essay_score['score'].unique()), 2):
    scores = sorted(train_essay_score['score'].unique())[idx:idx+2]
    sns.histplot(train_essay_score[train_essay_score['score'].isin(scores)]['Total_action_time'],
                 label=f'Score {scores}', kde=True)

plt.title('Distribution of Total_action_time for Each Score')
plt.xlabel('Total_action_time')
plt.ylabel('Frequency')

plt.legend()

plt.savefig('Distribution_of_Total_action_time_for_Each_Score.png', dpi=300)

plt.show()


# In[ ]:


sns.set(style="whitegrid")

plt.figure(figsize=(14, 7))

# Loop through unique scores and create a histogram for each
for idx in range(0, len(train_essay_score['score'].unique()), 2):
    scores = sorted(train_essay_score['score'].unique())[idx:idx+2]
    sns.histplot(train_essay_score[train_essay_score['score'].isin(scores)]['Num_Sentences'],
                 label=f'Score {scores}', kde=True)

plt.title('Distribution of Num_Sentences for Each Score')
plt.xlabel('Num_Sentences')
plt.ylabel('Frequency')

plt.legend()

plt.savefig('Distribution_of_Num_Sentences_for_Each_Score.png', dpi=300)

plt.show()


# In[ ]:


sns.set(style="whitegrid")

# Boxplot for Count_Positive_wait_time
plt.figure(figsize=(10, 6))
colors = sns.color_palette("husl", n_colors=1)
sns.boxplot(x='score', y='Count_Positive_wait_time', data=train_essay_score, palette=colors)

plt.title('Boxplot of Count_Positive_wait_time for Each Score')
plt.xlabel('Score')
plt.ylabel('Count_Positive_wait_time')

plt.savefig('Boxplot_of_Count_Positive_wait_time_for_Each_Score.png', dpi=300)

plt.show()

# Boxplot for Count_Negative_wait_time
plt.figure(figsize=(10, 6))
sns.boxplot(x='score', y='Count_Negative_wait_time', data=train_essay_score, palette=colors)

plt.title('Boxplot of Count_Negative_wait_time for Each Score')
plt.xlabel('Score')
plt.ylabel('Count_Negative_wait_time')

plt.savefig('Boxplot_of_Count_Negative_wait_time_for_Each_Score.png', dpi=300)

plt.show()


# In[ ]:


colors = sns.color_palette("husl", n_colors=1)

plt.figure(figsize=(10, 6))
sns.boxplot(x='score', y='Count_Positive_wait_time', data=train_essay_score, palette=colors)

plt.title('Boxplot of Count_Positive_wait_time for Each Score')
plt.xlabel('Score')
plt.ylabel('Count_Positive_wait_time')

plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='score', y='Count_Negative_wait_time', data=train_essay_score, palette=colors)

plt.title('Boxplot of Count_Negative_wait_time for Each Score')
plt.xlabel('Score')
plt.ylabel('Count_Negative_wait_time')

plt.show()


# In[ ]:


train_essay_score['score'].value_counts()


# In[ ]:


bins = [0, 2, 4, 6]
labels = ['0.5-2.0', '2.5-4.0', '4.5-6.0']

# Create a new column 'score_category' based on the specified bins
train_essay_score['3_score_category'] = pd.cut(train_essay_score['score'], bins=bins, labels=labels, include_lowest=True)

target = 'score'
train_essay_score_ordered = pd.concat([train_essay_score.drop(target,axis=1), train_essay_score[target]],axis=1)


# In[ ]:


train_essay_score_ordered.head()


# In[ ]:


# Define a colormap
royalblue = LinearSegmentedColormap.from_list('royalblue', [(0, (1,1,1)), (1, (0.25,0.41,0.88))])
royalblue_r = royalblue.reversed()

numeric_columns = train_essay_score.select_dtypes(include=np.number).columns
train_essay_score_numeric = train_essay_score[numeric_columns]

train_essay_score_numeric_ordered = pd.concat([train_essay_score_numeric.drop(target,axis=1), train_essay_score_numeric[target]],axis=1)

corr = train_essay_score_numeric_ordered.corr(method='spearman')

# Create a mask so that we see the correlation values only once
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask,1)] = True

# Plot the heatmap correlation
plt.figure(figsize=(12,8), dpi=80)
sns.heatmap(corr, mask=mask, annot=True, cmap=royalblue, fmt='.2f', linewidths=0.2)

plt.savefig('Correlation_Heatmap.png', dpi=300)

plt.show()


# In[ ]:


train_essay_score_numeric_ordered.head()


# In[ ]:


train_essay_score_ordered.select_dtypes(include=np.number).columns.tolist()


# In[ ]:


sns.set_palette(['royalblue', 'darkturquoise', 'limegreen'])

Num_Features = train_essay_score_ordered.select_dtypes(include=np.number).columns.tolist()
Num_Features.remove('score')

fig, ax = plt.subplots(len(Num_Features), 2, figsize=(30,75), dpi=200, gridspec_kw={'width_ratios': [1, 2]})
Target = '3_score_category'


for i,col in enumerate(Num_Features):
    # barplot
    graph = sns.barplot(data=train_essay_score_ordered, x=Target, y=col, ax=ax[i,0])
    # kde Plot
    sns.kdeplot(data=train_essay_score_ordered[train_essay_score_ordered[Target]=='0.5-2.0'], x=col, fill=True, linewidth=2, ax=ax[i,1], label='0.5-2.0')
    sns.kdeplot(data=train_essay_score_ordered[train_essay_score_ordered[Target]=='2.5-4.0'], x=col, fill=True, linewidth=2, ax=ax[i,1], label='2.5-4.0')
    sns.kdeplot(data=train_essay_score_ordered[train_essay_score_ordered[Target]=='4.5-6.0'], x=col, fill=True, linewidth=2, ax=ax[i,1], label='4.5-6.0')
    ax[i,1].set_yticks([])
    ax[i,1].legend(title='3_score_category', loc='upper right')
    # Add bar sizes to our plot
    for cont in graph.containers:
        graph.bar_label(cont, fmt='         %.3g')

plt.suptitle('Numerical Features vs Target Distribution', fontsize=22)
plt.tight_layout()
plt.savefig('Kdeplots.png', dpi=300)
plt.show()


# In[ ]:


import math

def round_to_nearest_half(value):
    if isinstance(value, np.ndarray):
        return np.round(value * 2) / 2
    else:
        return round(value * 2) / 2


# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, classification_report

Target = 'score'
X = train_essay_score_numeric_ordered.drop(Target, axis=1)
y = train_essay_score_numeric_ordered[Target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=141, stratify=y)
train = pd.concat([X_train, y_train], axis=1)

scaler = StandardScaler()


# In[ ]:


train.shape


# In[ ]:


y_train.value_counts()


# In[ ]:


def augment_data(X, y, num_samples, noise_factor=0.25):
    Features = X.columns
    X = scaler.fit_transform(X)
    X_augmented_data = pd.DataFrame(X, columns=Features)  # Convert X to DataFrame
    y_augmented_data = pd.Series(y.iloc[:, 0])  # Convert y to Series

    for _ in range(num_samples):
        X_augmented = X + noise_factor * np.random.normal(size=X.shape)
        y_augmented = y + noise_factor * np.random.normal(size=y.shape)

        X_augmented_df = pd.DataFrame(X_augmented, columns=Features)
        
        X_augmented_data = pd.concat([X_augmented_data, X_augmented_df], axis=0, ignore_index=True)
        y_augmented_data = pd.concat([pd.DataFrame(y_augmented_data), y_augmented])
        
    X_augmented_data.reset_index(drop=True, inplace=True)
    y_augmented_data.reset_index(drop=True, inplace=True)
    aug_data = pd.concat([X_augmented_data, y_augmented_data], axis=1)

    return pd.concat([X_augmented_data, y_augmented_data], axis=1)


# In[ ]:


Max_data_point = max(train['score'].value_counts())
train_aug = pd.DataFrame()
Target = 'score'

for score in train['score'].unique():
    data_len = train[train['score'] == score].shape[0]
    num_samples = round(Max_data_point / data_len) - 1
    X = train[train['score'] == score].drop(Target, axis=1)
    y = train[train['score'] == score][[Target]]
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    result_df = augment_data(X, y, num_samples, noise_factor=0.05)

    train_aug = pd.concat([train_aug, result_df])


# In[ ]:


train_aug.shape


# In[ ]:


train_aug.tail()


# In[ ]:


train_aug['score'] = train_aug['score'].apply(round_to_nearest_half)
train_aug.tail()


# In[ ]:


X_train = train_aug.drop(Target, axis=1)
y_train = train_aug[Target]


# In[ ]:


y_train.value_counts()


# In[ ]:


X_test = scaler.transform(X_test)

# Define the model with dropout layers
model = Sequential()
model.add(Dense(2048, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(2048, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(1024, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(512, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(256, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))  
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))  
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model with the weighted loss function
model.compile(optimizer='rmsprop', loss='mean_squared_error')

# Define callbacks for early stopping and model checkpointing
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Train the model with callbacks
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, model_checkpoint]
)

# Load the best model from model checkpoint
best_model = load_model('best_model.h5')
# best_model = load_model('best_model.h5', custom_objects={'loss': weighted_loss})

# Make predictions using the best model
predictions = best_model.predict(X_test)

predictions = np.clip(predictions, 0.3, 6.2)


predictions_rounded_values = [round_to_nearest_half(value) for value in predictions]

# Print some predictions
for true_value, pred_value in zip(y_test[:5], predictions_rounded_values[:5]):
    print(f'True Value: {true_value:.2f}, Predicted Value: {pred_value[0]}')


# In[ ]:


# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'R-squared (R2): {r2:.4f}')

# Plot the training history (loss)
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


for true_value, pred_value in zip(y_test[:5], predictions[:5]):
    print(f'True Value: {true_value:.2f}, Predicted Value: {pred_value[0]:.2f}')


# In[ ]:


predictions_rounded_values = [round_to_nearest_half(value) for value in predictions]

for true_value, pred_value in zip(y_test[:10], predictions_rounded_values[:10]):
    print(f'True Value: {true_value:.2f}, Predicted Value: {pred_value[0]}')


# In[ ]:


np.unique(predictions_rounded_values, return_counts=True)


# In[ ]:


np.unique(y_test, return_counts=True)


# In[ ]:


def feature_extraction(df):
    df['wait_time'] = df['down_time'] - df.groupby('id')['up_time'].shift()
    df['wait_time'].fillna(0, inplace=True)
    
    wait_time_df = df.groupby('id')['wait_time'].agg(
    Total_Positive_wait_time=lambda x: x[x > 0].sum(),
    Total_Negative_wait_time=lambda x: x[x < 0].sum(),
    Count_Positive_wait_time=lambda x: (x > 0).sum(),
    Count_Negative_wait_time=lambda x: (x < 0).sum()
    ).reset_index()
    
    action_time_df = df.groupby('id')['action_time'].agg(
    Total_action_time=lambda x: x[x > 0].sum()
    ).reset_index()
    
    df['Updated_activity'] = df['activity'].apply(update_action)
    
    action_counts = df.groupby('id')['Updated_activity'].value_counts().unstack(fill_value=0)


    essayConstructor=EssayConstructor()
    essay=essayConstructor.getEssays(df)
    
    essay['word_count'] = essay['essay'].apply(lambda x: count_words(x))
    
    essay = pd.merge(essay, wait_time_df, on='id', how='left')
    essay = pd.merge(essay, action_time_df, on='id', how='left')
    essay = pd.merge(essay, action_counts, on='id', how='left')       

    essay['Num_Sentences'] = essay['essay'].apply(count_sentences)
    essay['Num_Paragraphs'] = essay['essay'].apply(count_paragraphs)
    
    essay_id = essay[['id']]
    
    required_columns = ['word_count', 'All Move From', 'Input', 'Nonproduction', 'Paste',
       'Remove/Cut', 'Replace', 'Total_Positive_wait_time',
       'Total_Negative_wait_time', 'Count_Positive_wait_time',
       'Count_Negative_wait_time', 'Total_action_time', 'Num_Sentences',
       'Num_Paragraphs']
    
    for col in required_columns:
        if col not in essay.columns:
            essay[col] = 0

    return essay_id, essay[required_columns]


# In[ ]:


processed_test_df_id, processed_test_df = feature_extraction(test_logs)


# In[ ]:


X_test = scaler.fit_transform(processed_test_df)
predictions = model.predict(X_test)
predictions


# In[ ]:


predictions_rounded_values = [round_to_nearest_half(value[0]) for value in predictions]


# In[ ]:


processed_test_df_id['score'] = predictions_rounded_values


# In[ ]:


processed_test_df_id.to_csv("submission.csv", index=False)


# In[ ]:


processed_test_df_id

