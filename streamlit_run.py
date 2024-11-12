import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
import seaborn as sns
import streamlit as st

import pyarrow as pa
from datasets import Dataset
from datasets import load_dataset
from datasets import DatasetDict


from transformers import AutoTokenizer

import torch
import torch.nn.functional as F
from transformers import AutoModel

from sklearn.metrics import f1_score,classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score,f1_score
from transformers import Trainer
from torch.nn.functional import cross_entropy
from transformers import pipeline

###########################################################################

def tokenize(batch,tokenizer):
    '''This function applies tokenizer to a batch of examples; padding=True will pad the examples with
    zeros to the size of the longest one in a batch, and truncation=True will truncate the examples
    to the model's maximum context size.'''
    return tokenizer(batch["Claim Description"],padding=True, truncation = True)


def filter_high_frequency_classes(data,col,pct):
    # Get value counts for each category in 'Coverage Code'
    coverage_counts =data[col].value_counts(normalize=True)*100
    
    # Filter to retain only categories with counts greater than 1 pct
    high_frequency_class = coverage_counts[coverage_counts > pct].index
    
    return high_frequency_class


def assign_others(cat,high_frequency_cat):
    if cat in high_frequency_cat:
        return cat
    else:
        return 'other'
        

def label_int2str(x,label_mapping):
    label_mapping_rev={}
    for k,v in label_mapping.items():
        label_mapping_rev[v]=k
    return label_mapping_rev[x]

def return_predictions(x,label_mapping):
    x = int(x['label'].split('_')[-1])
    return label_int2str(x,label_mapping)


def plot_confusion_matrix(y_preds,y_true, labels=None):
    labels = y_preds.unique()
    labels.sort()
    cm = confusion_matrix(y_true, y_preds, normalize="true",labels=labels)
    fig,ax = plt.subplots(figsize=(20,15))
    disp = ConfusionMatrixDisplay(confusion_matrix =cm,display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.xticks(rotation=90)
    plt.title("Normalized Confusion Matrix")
    plt.savefig('Normalized Confusion Matrix')
    # plt.show()

    return fig

def str2int(x,label_mapping):
    return label_mapping[x]


###########################################################################


# Define the function to load and preprocess the data, then run the model
def execute_model(data,task):

    df = data
    df = df[df['Claim Description'].notna()]

    if task==1: # Coverage Code
        model_id = "abhxaxhbshxahxn/CoverageCodePred"
        pct = 1
        col = 'Coverage Code'
        pred_col = 'Coverage Code Merged'


        coverage_code_label_mapping = {'AB': 0, 'AD': 1, 'AL': 2,'AP': 3, 'GB': 4, 'GD': 5, 'NS': 6, 'PA': 7, 'PB': 8, 'RB': 9, 'other': 10}

        high_frequency_coverage_code = coverage_code_label_mapping.keys()
        df['Coverage Code Merged'] = df['Coverage Code'].apply(lambda x:assign_others(x,high_frequency_coverage_code))

        df['Coverage Code Encoded'] = df['Coverage Code Merged'].apply(lambda x:str2int(x,coverage_code_label_mapping))



        use_encodings = coverage_code_label_mapping
    
    else:
        model_id = "abhxaxhbshxahxn/AccidentSource_Predictions"
        col ='Accident Source'
        pred_col = 'Accident Source Merged'
        pct = 1
        
        accident_source_label_mapping = {'Alleged Negligent Act': 0,
            'Alleged contamination or spoilage': 1,
            'Alleged damage to property of others': 2,
            'Alleged design flaw, defect': 3,
            'Alleged foreign object in product': 4,
            'Alleged improper maintenance - other': 5,
            'Backed into vehicle or object': 6,
            'Cart': 7,
            'Ground/floor': 8,
            'Human Action, NOC': 9,
            'Intersection accident': 10,
            'Not Otherwise Classified': 11,
            'Our vehicle struck in rear': 12,
            'Pothole': 13,
            'Sideswipe or lane change': 14,
            'Struck animal or object': 15,
            'Struck parked vehicle': 16,
            'Struck vehicle in rear': 17,
            'Struck/pulled down wires': 18,
            'Vehicle Accident': 19,
            'Windshield': 20,
            'e-Commerce': 21,
            'other': 22}
        
        high_frequency_accident_source = accident_source_label_mapping.keys()

        df['Accident Source Merged'] = df['Accident Source'].apply(lambda x:assign_others(x,high_frequency_accident_source))

        df['Accident Source Encoded'] = df['Accident Source Merged'].apply(lambda x:str2int(x,accident_source_label_mapping))


        use_encodings = accident_source_label_mapping



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:",device)
    # model = AutoModelForSequenceClassification.from_pretrained(model_path, return_dict=True).to(device)
    # tokenizer = AutoTokenizer.from_pretrained(model_path)


    classifier_ = pipeline("text-classification",model= model_id,device=device)


    

    custom_input = [df.loc[idx,'Claim Description'] for idx in df.index]
    correct = [df.loc[idx,pred_col] for idx in df.index]
    preds = classifier_(custom_input, top_k=1)
    pred_df = pd.DataFrame(preds)

    pred_df['Claim Description'] = custom_input
    pred_df['Actual'] = correct
    pred_df['Predictions'] = pred_df[0].apply(lambda x:return_predictions(x,label_mapping=use_encodings))
    pred_df['Confidence'] = pred_df[0].apply(lambda x:x['score'])
    pred_df.drop(0,axis=1,inplace=True)


    # plot_confusion_matrix(pred_df['Actual'], pred_df['Predictions'],use_encodings.keys())


    return pred_df,use_encodings



###########################################################################

# Streamlit App
def main():
    st.title("Text Classification Model Prediction")

    # Load Data Section
    uploaded_file = st.file_uploader("Upload a CSV file for text classification", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data Loaded Successfully")
        st.write(data.head())  # Display the first few rows of the data

     # Task Selection Section
    task = st.radio(
        "Select the task for prediction: Enter 1 for Coverage Code or 2 for Accidental Source",
        (1, 2)
    )

    # Run Model Section
    if st.button("Run"):
        if uploaded_file is not None:
            st.write("Running Model...")
            results,use_encodings = execute_model(data,task)

            st.write("Predictions:")
            st.write(results)

            st.write('Displaying Normalized Confusion Matrix')
            fig = plot_confusion_matrix(results['Actual'], results['Predictions'])
            st.pyplot(fig)

            # Generate the classification report as a dictionary
            cls_report_dict = classification_report(results['Actual'], results['Predictions'], output_dict=True)
            # Convert to DataFrame
            cls_report_df = pd.DataFrame(cls_report_dict).transpose()
            st.write("Classification Report:")
            st.write(cls_report_df)

            if task==1:
                file_name = 'CoverageCode_Predictions.csv'
            else:
                file_name = 'AccidentSource_Predictions.csv'

            # Optionally, save the results to a file
            results.to_csv(file_name, index=False)
            st.write(f"Results saved as {file_name}")
        else:
            st.write("Please upload a CSV file before running the model.")

if __name__ == "__main__":
    main()







