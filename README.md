# LendingClub-NeuralNetwork

## Objective

The objective of this project is to design an Artificial Neural Network (ANN) model that accurately predicts whether an individual will default on a loan.  The neural network will model the risk of a potential borrower based on the various attributes of the borrower and the loan.  Since this is a binary classification problem, the output will be either a 0 to indicate the “No Default” class and a 1 to indicate the “Default” class.  The model will also produce class probabilities to give the probability of default as a percentage.  We will consider loans with the status of “Charged Off” or “Default” as defaulted loans – there are 43,838 instances of this class.  There are 200,202 instances of fully paid loans, bringing the total records to 244,040‬ after the data was cleaned.

## Data Source and Characteristics

[Link to Data Source](https:/www.kaggle.com/wendykan/lending-club-loan-data)

The data used for this model is from Kaggle and sourced from Lending Club – a peer-to-peer lending company – containing complete loan data for all loans issued through the 2007-2015.  Credit risk modelling has been on the rise in financial industries as firms are using the data available to minimize their risk while maximizing their returns.  Investors on the Lending Club platform need to properly assess credit risk to reduce the likelihood of losses from default and delayed repayment.  Using the data available to build an AI model, we can derive insights into loans to maximize ROI.  There are many redundant and unnecessary columns so we will need to remove them from the data set – and after the cleansing process we are left with a total of 244,040 records of fully paid and defaulted loans. 
