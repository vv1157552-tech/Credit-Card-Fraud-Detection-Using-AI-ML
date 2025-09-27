#!/usr/bin/env python3
"""
Credit Card Fraud Detection - Synthetic Dataset Generation
This script generates a realistic synthetic credit card fraud dataset
for machine learning model development and testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from sklearn.preprocessing import StandardScaler
import os

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

def generate_synthetic_fraud_dataset(n_samples=100000, fraud_rate=0.002):
    """
    Generate a synthetic credit card fraud dataset with realistic features.
    
    Parameters:
    n_samples (int): Total number of transactions to generate
    fraud_rate (float): Proportion of fraudulent transactions (default: 0.2%)
    
    Returns:
    pandas.DataFrame: Generated dataset with features and labels
    """
    
    print(f"Generating {n_samples} synthetic credit card transactions...")
    print(f"Fraud rate: {fraud_rate*100:.2f}%")
    
    # Calculate number of fraud and legitimate transactions
    n_fraud = int(n_samples * fraud_rate)
    n_legitimate = n_samples - n_fraud
    
    # Initialize lists to store features
    data = []
    
    # Generate legitimate transactions
    print("Generating legitimate transactions...")
    for i in range(n_legitimate):
        # Time features (probabilities sum to 1.0)
        hour_probs = [0.02, 0.01, 0.01, 0.01, 0.01, 0.02, 
                      0.04, 0.06, 0.08, 0.09, 0.08, 0.07,
                      0.06, 0.05, 0.04, 0.03, 0.04, 0.05,
                      0.06, 0.07, 0.08, 0.06, 0.04, 0.03]
        hour_probs = np.array(hour_probs) / np.sum(hour_probs)  # Normalize to sum to 1
        hour = np.random.choice(range(24), p=hour_probs)
        day_of_week = np.random.randint(0, 7)
        
        # Transaction amount (log-normal distribution for realistic amounts)
        amount = np.random.lognormal(mean=3.5, sigma=1.2)
        amount = max(1.0, min(amount, 5000.0))  # Cap between $1 and $5000
        
        # Merchant category (0-17 representing different business types)
        merchant_probs = [0.15, 0.12, 0.10, 0.08, 0.07, 0.06,
                          0.05, 0.05, 0.04, 0.04, 0.04, 0.03,
                          0.03, 0.03, 0.03, 0.03, 0.02, 0.02]
        merchant_probs = np.array(merchant_probs) / np.sum(merchant_probs)
        merchant_category = np.random.choice(range(18), p=merchant_probs)
        
        # Customer features
        customer_age = np.random.normal(45, 15)
        customer_age = max(18, min(customer_age, 80))
        
        # Account age in days
        account_age = np.random.exponential(scale=365*2)  # Average 2 years
        account_age = min(account_age, 365*10)  # Max 10 years
        
        # Transaction frequency features
        transactions_last_day = np.random.poisson(lam=2)
        transactions_last_week = np.random.poisson(lam=8)
        transactions_last_month = np.random.poisson(lam=25)
        
        # Amount-based features
        avg_amount_last_month = np.random.lognormal(mean=3.2, sigma=0.8)
        max_amount_last_month = avg_amount_last_month * np.random.uniform(1.5, 3.0)
        
        # Location features (simplified as distance from home)
        distance_from_home = np.random.exponential(scale=10)  # km
        distance_from_last_transaction = np.random.exponential(scale=5)  # km
        
        # Device and channel features
        online_transaction = np.random.choice([0, 1], p=[0.6, 0.4])
        mobile_transaction = np.random.choice([0, 1], p=[0.7, 0.3])
        
        # Risk indicators (lower for legitimate transactions)
        velocity_risk = np.random.beta(2, 8)  # Skewed towards low risk
        amount_risk = np.random.beta(2, 8)
        location_risk = np.random.beta(2, 8)
        
        # Create transaction record
        transaction = {
            'Time': hour + (day_of_week * 24),
            'V1': np.random.normal(0, 1),  # PCA components (anonymized)
            'V2': np.random.normal(0, 1),
            'V3': np.random.normal(0, 1),
            'V4': np.random.normal(0, 1),
            'V5': np.random.normal(0, 1),
            'V6': np.random.normal(0, 1),
            'V7': np.random.normal(0, 1),
            'V8': np.random.normal(0, 1),
            'V9': np.random.normal(0, 1),
            'V10': np.random.normal(0, 1),
            'V11': np.random.normal(0, 1),
            'V12': np.random.normal(0, 1),
            'V13': np.random.normal(0, 1),
            'V14': np.random.normal(0, 1),
            'V15': np.random.normal(0, 1),
            'V16': np.random.normal(0, 1),
            'V17': np.random.normal(0, 1),
            'V18': np.random.normal(0, 1),
            'V19': np.random.normal(0, 1),
            'V20': np.random.normal(0, 1),
            'V21': np.random.normal(0, 1),
            'V22': np.random.normal(0, 1),
            'V23': np.random.normal(0, 1),
            'V24': np.random.normal(0, 1),
            'V25': np.random.normal(0, 1),
            'V26': np.random.normal(0, 1),
            'V27': np.random.normal(0, 1),
            'V28': np.random.normal(0, 1),
            'Amount': amount,
            'Hour': hour,
            'Day_of_Week': day_of_week,
            'Merchant_Category': merchant_category,
            'Customer_Age': customer_age,
            'Account_Age': account_age,
            'Transactions_Last_Day': transactions_last_day,
            'Transactions_Last_Week': transactions_last_week,
            'Transactions_Last_Month': transactions_last_month,
            'Avg_Amount_Last_Month': avg_amount_last_month,
            'Max_Amount_Last_Month': max_amount_last_month,
            'Distance_From_Home': distance_from_home,
            'Distance_From_Last_Transaction': distance_from_last_transaction,
            'Online_Transaction': online_transaction,
            'Mobile_Transaction': mobile_transaction,
            'Velocity_Risk': velocity_risk,
            'Amount_Risk': amount_risk,
            'Location_Risk': location_risk,
            'Class': 0  # Legitimate transaction
        }
        
        data.append(transaction)
    
    # Generate fraudulent transactions
    print("Generating fraudulent transactions...")
    for i in range(n_fraud):
        # Fraudulent transactions often occur at unusual times
        fraud_hour_probs = [0.08, 0.08, 0.08, 0.08, 0.08, 0.08,
                            0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
                            0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
                            0.02, 0.02, 0.08, 0.08, 0.08, 0.08]
        fraud_hour_probs = np.array(fraud_hour_probs) / np.sum(fraud_hour_probs)
        hour = np.random.choice(range(24), p=fraud_hour_probs)
        day_of_week = np.random.randint(0, 7)
        
        # Fraudulent amounts tend to be higher or very specific
        if np.random.random() < 0.3:
            # Small test transactions
            amount = np.random.uniform(1, 10)
        else:
            # Large fraudulent transactions
            amount = np.random.lognormal(mean=5.0, sigma=1.5)
            amount = max(100.0, min(amount, 10000.0))
        
        # Different merchant category distribution for fraud
        fraud_merchant_probs = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                                0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                                0.05, 0.05, 0.05, 0.05, 0.15, 0.15]
        fraud_merchant_probs = np.array(fraud_merchant_probs) / np.sum(fraud_merchant_probs)
        merchant_category = np.random.choice(range(18), p=fraud_merchant_probs)
        
        # Customer features (may be stolen account info)
        customer_age = np.random.normal(40, 20)
        customer_age = max(18, min(customer_age, 80))
        
        # Account age
        account_age = np.random.exponential(scale=365*1.5)
        account_age = min(account_age, 365*10)
        
        # Higher transaction frequency for fraud
        transactions_last_day = np.random.poisson(lam=5)
        transactions_last_week = np.random.poisson(lam=15)
        transactions_last_month = np.random.poisson(lam=40)
        
        # Amount features
        avg_amount_last_month = np.random.lognormal(mean=3.0, sigma=1.0)
        max_amount_last_month = avg_amount_last_month * np.random.uniform(2.0, 5.0)
        
        # Location features (often far from home for fraud)
        distance_from_home = np.random.exponential(scale=50)  # km
        distance_from_last_transaction = np.random.exponential(scale=25)  # km
        
        # Device and channel features
        online_transaction = np.random.choice([0, 1], p=[0.3, 0.7])  # More online fraud
        mobile_transaction = np.random.choice([0, 1], p=[0.5, 0.5])
        
        # Higher risk indicators for fraudulent transactions
        velocity_risk = np.random.beta(8, 2)  # Skewed towards high risk
        amount_risk = np.random.beta(6, 3)
        location_risk = np.random.beta(7, 2)
        
        # Create fraudulent transaction record
        transaction = {
            'Time': hour + (day_of_week * 24),
            'V1': np.random.normal(2, 1.5),  # Different distribution for fraud
            'V2': np.random.normal(-1, 1.2),
            'V3': np.random.normal(1.5, 1.3),
            'V4': np.random.normal(-0.5, 1.1),
            'V5': np.random.normal(0.8, 1.4),
            'V6': np.random.normal(-1.2, 1.2),
            'V7': np.random.normal(1.1, 1.3),
            'V8': np.random.normal(-0.8, 1.1),
            'V9': np.random.normal(1.3, 1.4),
            'V10': np.random.normal(-1.1, 1.2),
            'V11': np.random.normal(0.9, 1.3),
            'V12': np.random.normal(-1.3, 1.1),
            'V13': np.random.normal(1.2, 1.4),
            'V14': np.random.normal(-0.9, 1.2),
            'V15': np.random.normal(1.4, 1.3),
            'V16': np.random.normal(-1.4, 1.1),
            'V17': np.random.normal(1.0, 1.4),
            'V18': np.random.normal(-1.0, 1.2),
            'V19': np.random.normal(1.5, 1.3),
            'V20': np.random.normal(-1.5, 1.1),
            'V21': np.random.normal(1.1, 1.4),
            'V22': np.random.normal(-1.1, 1.2),
            'V23': np.random.normal(1.3, 1.3),
            'V24': np.random.normal(-1.3, 1.1),
            'V25': np.random.normal(1.2, 1.4),
            'V26': np.random.normal(-1.2, 1.2),
            'V27': np.random.normal(1.4, 1.3),
            'V28': np.random.normal(-1.4, 1.1),
            'Amount': amount,
            'Hour': hour,
            'Day_of_Week': day_of_week,
            'Merchant_Category': merchant_category,
            'Customer_Age': customer_age,
            'Account_Age': account_age,
            'Transactions_Last_Day': transactions_last_day,
            'Transactions_Last_Week': transactions_last_week,
            'Transactions_Last_Month': transactions_last_month,
            'Avg_Amount_Last_Month': avg_amount_last_month,
            'Max_Amount_Last_Month': max_amount_last_month,
            'Distance_From_Home': distance_from_home,
            'Distance_From_Last_Transaction': distance_from_last_transaction,
            'Online_Transaction': online_transaction,
            'Mobile_Transaction': mobile_transaction,
            'Velocity_Risk': velocity_risk,
            'Amount_Risk': amount_risk,
            'Location_Risk': location_risk,
            'Class': 1  # Fraudulent transaction
        }
        
        data.append(transaction)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    print(f"Dataset generated successfully!")
    print(f"Total transactions: {len(df)}")
    print(f"Fraudulent transactions: {df['Class'].sum()}")
    print(f"Legitimate transactions: {len(df) - df['Class'].sum()}")
    print(f"Fraud rate: {df['Class'].mean()*100:.3f}%")
    
    return df

def save_dataset(df, filename='credit_card_fraud_dataset.csv'):
    """Save the dataset to a CSV file."""
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")
    
    # Print basic statistics
    print("\nDataset Statistics:")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("\nFeature types:")
    print(df.dtypes.value_counts())
    
    print("\nClass distribution:")
    print(df['Class'].value_counts())
    print(df['Class'].value_counts(normalize=True))

if __name__ == "__main__":
    # Generate the dataset
    dataset = generate_synthetic_fraud_dataset(n_samples=100000, fraud_rate=0.002)
    
    # Save to file
    save_dataset(dataset, '/home/ubuntu/credit_card_fraud_dataset.csv')
    
    print("\nDataset generation completed successfully!")

