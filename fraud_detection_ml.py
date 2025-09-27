#!/usr/bin/env python3
"""
Credit Card Fraud Detection - Machine Learning Implementation
This script implements a comprehensive fraud detection system using multiple ML algorithms.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, f1_score, precision_score, 
                           recall_score, accuracy_score)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import warnings
import joblib
import os
from datetime import datetime

warnings.filterwarnings('ignore')

class FraudDetectionSystem:
    """
    Comprehensive Credit Card Fraud Detection System
    """
    
    def __init__(self, data_path='credit_card_fraud_dataset.csv'):
        """Initialize the fraud detection system."""
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.scaler = None
        
        print("Fraud Detection System Initialized")
        print("=" * 50)
    
    def load_and_explore_data(self):
        """Load and perform initial exploration of the dataset."""
        print("Loading dataset...")
        self.data = pd.read_csv(self.data_path)
        
        print(f"Dataset loaded successfully!")
        print(f"Shape: {self.data.shape}")
        print(f"Memory usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Basic statistics
        print("\nDataset Overview:")
        print(self.data.info())
        
        print("\nClass Distribution:")
        class_counts = self.data['Class'].value_counts()
        print(class_counts)
        print(f"Fraud rate: {self.data['Class'].mean()*100:.3f}%")
        
        # Check for missing values
        print("\nMissing Values:")
        missing_values = self.data.isnull().sum()
        if missing_values.sum() == 0:
            print("No missing values found.")
        else:
            print(missing_values[missing_values > 0])
        
        # Statistical summary
        print("\nStatistical Summary:")
        print(self.data.describe())
        
        return self.data
    
    def visualize_data_distribution(self):
        """Create visualizations for data exploration."""
        print("Creating data distribution visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Credit Card Fraud Detection - Data Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Class distribution
        class_counts = self.data['Class'].value_counts()
        axes[0, 0].pie(class_counts.values, labels=['Legitimate', 'Fraud'], autopct='%1.2f%%', 
                       colors=['lightblue', 'lightcoral'])
        axes[0, 0].set_title('Class Distribution')
        
        # 2. Amount distribution by class
        legitimate = self.data[self.data['Class'] == 0]['Amount']
        fraud = self.data[self.data['Class'] == 1]['Amount']
        
        axes[0, 1].hist(legitimate, bins=50, alpha=0.7, label='Legitimate', color='blue', density=True)
        axes[0, 1].hist(fraud, bins=50, alpha=0.7, label='Fraud', color='red', density=True)
        axes[0, 1].set_xlabel('Transaction Amount')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Amount Distribution by Class')
        axes[0, 1].legend()
        axes[0, 1].set_xlim(0, 1000)  # Focus on lower amounts for better visibility
        
        # 3. Time distribution
        axes[0, 2].hist(self.data[self.data['Class'] == 0]['Hour'], bins=24, alpha=0.7, 
                        label='Legitimate', color='blue', density=True)
        axes[0, 2].hist(self.data[self.data['Class'] == 1]['Hour'], bins=24, alpha=0.7, 
                        label='Fraud', color='red', density=True)
        axes[0, 2].set_xlabel('Hour of Day')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].set_title('Transaction Time Distribution')
        axes[0, 2].legend()
        
        # 4. Correlation heatmap (subset of features)
        correlation_features = ['Amount', 'Hour', 'Customer_Age', 'Account_Age', 
                               'Velocity_Risk', 'Amount_Risk', 'Location_Risk', 'Class']
        corr_matrix = self.data[correlation_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                    ax=axes[1, 0], fmt='.2f')
        axes[1, 0].set_title('Feature Correlation Matrix')
        
        # 5. Risk indicators by class
        risk_features = ['Velocity_Risk', 'Amount_Risk', 'Location_Risk']
        risk_data = []
        for feature in risk_features:
            for class_val in [0, 1]:
                values = self.data[self.data['Class'] == class_val][feature]
                risk_data.extend([(feature, 'Legitimate' if class_val == 0 else 'Fraud', val) 
                                 for val in values])
        
        risk_df = pd.DataFrame(risk_data, columns=['Risk_Type', 'Class', 'Value'])
        sns.boxplot(data=risk_df, x='Risk_Type', y='Value', hue='Class', ax=axes[1, 1])
        axes[1, 1].set_title('Risk Indicators by Class')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Feature importance preview (using simple correlation)
        feature_importance = abs(self.data.corr()['Class']).sort_values(ascending=False)[1:11]
        axes[1, 2].barh(range(len(feature_importance)), feature_importance.values)
        axes[1, 2].set_yticks(range(len(feature_importance)))
        axes[1, 2].set_yticklabels(feature_importance.index)
        axes[1, 2].set_xlabel('Absolute Correlation with Fraud')
        axes[1, 2].set_title('Top 10 Features by Correlation')
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/data_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Data distribution visualizations saved to 'data_distribution_analysis.png'")
    
    def preprocess_data(self):
        """Preprocess the data for machine learning."""
        print("Preprocessing data...")
        
        # Separate features and target
        X = self.data.drop('Class', axis=1)
        y = self.data['Class']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        print(f"Training fraud rate: {self.y_train.mean()*100:.3f}%")
        print(f"Test fraud rate: {self.y_test.mean()*100:.3f}%")
        
        # Scale the features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Data preprocessing completed.")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_models(self):
        """Train multiple machine learning models."""
        print("Training machine learning models...")
        
        # Define models
        models_config = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'SVM': SVC(probability=True, random_state=42),
            'Isolation Forest': IsolationForest(contamination=0.002, random_state=42)
        }
        
        # Train each model
        for name, model in models_config.items():
            print(f"Training {name}...")
            
            if name == 'Isolation Forest':
                # Unsupervised model - train on features only
                model.fit(self.X_train_scaled)
                # Predict anomalies (-1 for outliers, 1 for inliers)
                train_pred = model.predict(self.X_train_scaled)
                test_pred = model.predict(self.X_test_scaled)
                
                # Convert to binary classification (1 for fraud, 0 for legitimate)
                train_pred_binary = np.where(train_pred == -1, 1, 0)
                test_pred_binary = np.where(test_pred == -1, 1, 0)
                
                self.models[name] = {
                    'model': model,
                    'train_predictions': train_pred_binary,
                    'test_predictions': test_pred_binary,
                    'train_probabilities': None,  # Isolation Forest doesn't provide probabilities
                    'test_probabilities': None
                }
            else:
                # Supervised models
                if name in ['Random Forest', 'XGBoost']:
                    # Use SMOTE for tree-based models to handle imbalance
                    smote = SMOTE(random_state=42)
                    X_train_balanced, y_train_balanced = smote.fit_resample(self.X_train_scaled, self.y_train)
                    model.fit(X_train_balanced, y_train_balanced)
                else:
                    # Train on original data for other models
                    model.fit(self.X_train_scaled, self.y_train)
                
                # Make predictions
                train_pred = model.predict(self.X_train_scaled)
                test_pred = model.predict(self.X_test_scaled)
                train_prob = model.predict_proba(self.X_train_scaled)[:, 1]
                test_prob = model.predict_proba(self.X_test_scaled)[:, 1]
                
                self.models[name] = {
                    'model': model,
                    'train_predictions': train_pred,
                    'test_predictions': test_pred,
                    'train_probabilities': train_prob,
                    'test_probabilities': test_prob
                }
        
        print("All models trained successfully!")
        return self.models
    
    def evaluate_models(self):
        """Evaluate all trained models."""
        print("Evaluating model performance...")
        
        self.results = {}
        
        for name, model_data in self.models.items():
            print(f"\nEvaluating {name}...")
            
            test_pred = model_data['test_predictions']
            test_prob = model_data['test_probabilities']
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, test_pred)
            precision = precision_score(self.y_test, test_pred)
            recall = recall_score(self.y_test, test_pred)
            f1 = f1_score(self.y_test, test_pred)
            
            # ROC AUC (only for models with probabilities)
            if test_prob is not None:
                roc_auc = roc_auc_score(self.y_test, test_prob)
            else:
                roc_auc = None
            
            # Confusion matrix
            cm = confusion_matrix(self.y_test, test_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'confusion_matrix': cm,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn,
                'true_positives': tp,
                'test_predictions': test_pred,
                'test_probabilities': test_prob
            }
            
            # Print results
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            if roc_auc is not None:
                print(f"ROC AUC: {roc_auc:.4f}")
            print(f"Confusion Matrix:")
            print(f"TN: {tn}, FP: {fp}")
            print(f"FN: {fn}, TP: {tp}")
        
        return self.results
    
    def create_performance_visualizations(self):
        """Create comprehensive performance visualizations."""
        print("Creating performance visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Credit Card Fraud Detection - Model Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Model comparison metrics
        models_with_probs = {name: results for name, results in self.results.items() 
                           if results['roc_auc'] is not None}
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        model_names = list(models_with_probs.keys())
        
        metric_data = []
        for metric in metrics:
            values = [models_with_probs[name][metric] for name in model_names]
            metric_data.append(values)
        
        x = np.arange(len(model_names))
        width = 0.15
        
        for i, (metric, values) in enumerate(zip(metrics, metric_data)):
            axes[0, 0].bar(x + i*width, values, width, label=metric.replace('_', ' ').title())
        
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].set_xticks(x + width * 2)
        axes[0, 0].set_xticklabels(model_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 1)
        
        # 2. ROC Curves
        for name, results in models_with_probs.items():
            if results['test_probabilities'] is not None:
                fpr, tpr, _ = roc_curve(self.y_test, results['test_probabilities'])
                axes[0, 1].plot(fpr, tpr, label=f"{name} (AUC = {results['roc_auc']:.3f})")
        
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. Precision-Recall Curves
        for name, results in models_with_probs.items():
            if results['test_probabilities'] is not None:
                precision, recall, _ = precision_recall_curve(self.y_test, results['test_probabilities'])
                axes[0, 2].plot(recall, precision, label=f"{name}")
        
        axes[0, 2].set_xlabel('Recall')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].set_title('Precision-Recall Curves')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # 4. Confusion Matrix Heatmap (Best Model)
        best_model = max(models_with_probs.keys(), key=lambda x: models_with_probs[x]['f1_score'])
        cm = self.results[best_model]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title(f'Confusion Matrix - {best_model}')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # 5. Feature Importance (Random Forest)
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']['model']
            feature_names = self.X_train.columns
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[::-1][:15]  # Top 15 features
            
            axes[1, 1].barh(range(len(indices)), importances[indices])
            axes[1, 1].set_yticks(range(len(indices)))
            axes[1, 1].set_yticklabels([feature_names[i] for i in indices])
            axes[1, 1].set_xlabel('Feature Importance')
            axes[1, 1].set_title('Top 15 Feature Importances (Random Forest)')
        
        # 6. Error Analysis
        error_data = []
        for name, results in self.results.items():
            error_data.append([
                name,
                results['false_positives'],
                results['false_negatives'],
                results['true_positives'],
                results['true_negatives']
            ])
        
        error_df = pd.DataFrame(error_data, columns=['Model', 'False Positives', 'False Negatives', 
                                                   'True Positives', 'True Negatives'])
        
        x = np.arange(len(error_df))
        width = 0.2
        
        axes[1, 2].bar(x - width*1.5, error_df['False Positives'], width, label='False Positives', color='red', alpha=0.7)
        axes[1, 2].bar(x - width*0.5, error_df['False Negatives'], width, label='False Negatives', color='orange', alpha=0.7)
        axes[1, 2].bar(x + width*0.5, error_df['True Positives'], width, label='True Positives', color='green', alpha=0.7)
        axes[1, 2].bar(x + width*1.5, error_df['True Negatives'], width, label='True Negatives', color='blue', alpha=0.7)
        
        axes[1, 2].set_xlabel('Models')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title('Prediction Analysis')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(error_df['Model'], rotation=45)
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/model_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Performance visualizations saved to 'model_performance_analysis.png'")
    
    def generate_detailed_report(self):
        """Generate a detailed performance report."""
        print("Generating detailed performance report...")
        
        report = []
        report.append("CREDIT CARD FRAUD DETECTION - MODEL PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Dataset size: {self.data.shape[0]} transactions")
        report.append(f"Features: {self.data.shape[1] - 1}")
        report.append(f"Fraud rate: {self.data['Class'].mean()*100:.3f}%")
        report.append("")
        
        report.append("DATASET OVERVIEW")
        report.append("-" * 20)
        report.append(f"Training samples: {len(self.X_train)}")
        report.append(f"Test samples: {len(self.X_test)}")
        report.append(f"Training fraud cases: {self.y_train.sum()}")
        report.append(f"Test fraud cases: {self.y_test.sum()}")
        report.append("")
        
        report.append("MODEL PERFORMANCE SUMMARY")
        report.append("-" * 30)
        
        # Sort models by F1 score
        sorted_models = sorted(self.results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        
        for rank, (name, results) in enumerate(sorted_models, 1):
            report.append(f"\n{rank}. {name}")
            report.append(f"   Accuracy:  {results['accuracy']:.4f}")
            report.append(f"   Precision: {results['precision']:.4f}")
            report.append(f"   Recall:    {results['recall']:.4f}")
            report.append(f"   F1-Score:  {results['f1_score']:.4f}")
            if results['roc_auc'] is not None:
                report.append(f"   ROC AUC:   {results['roc_auc']:.4f}")
            
            # Confusion matrix details
            tn, fp, fn, tp = results['true_negatives'], results['false_positives'], results['false_negatives'], results['true_positives']
            report.append(f"   True Positives:  {tp}")
            report.append(f"   False Positives: {fp}")
            report.append(f"   True Negatives:  {tn}")
            report.append(f"   False Negatives: {fn}")
            
            # Business metrics
            if tp + fn > 0:
                fraud_detection_rate = tp / (tp + fn)
                report.append(f"   Fraud Detection Rate: {fraud_detection_rate:.4f}")
            
            if fp + tn > 0:
                false_positive_rate = fp / (fp + tn)
                report.append(f"   False Positive Rate:  {false_positive_rate:.4f}")
        
        report.append("\n" + "=" * 60)
        report.append("RECOMMENDATIONS")
        report.append("-" * 15)
        
        best_model = sorted_models[0][0]
        best_results = sorted_models[0][1]
        
        report.append(f"Best performing model: {best_model}")
        report.append(f"Recommended for production deployment based on F1-Score: {best_results['f1_score']:.4f}")
        
        if best_results['false_positives'] > best_results['true_positives']:
            report.append("WARNING: High false positive rate detected.")
            report.append("Consider adjusting classification threshold or additional feature engineering.")
        
        if best_results['false_negatives'] > best_results['true_positives'] * 0.1:
            report.append("WARNING: Significant false negative rate detected.")
            report.append("Consider ensemble methods or additional training data.")
        
        # Save report
        with open('/home/ubuntu/fraud_detection_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print("Detailed report saved to 'fraud_detection_report.txt'")
        return report
    
    def save_models(self):
        """Save trained models for future use."""
        print("Saving trained models...")
        
        # Create models directory
        os.makedirs('/home/ubuntu/models', exist_ok=True)
        
        # Save each model
        for name, model_data in self.models.items():
            model_filename = f"/home/ubuntu/models/{name.lower().replace(' ', '_')}_model.joblib"
            joblib.dump(model_data['model'], model_filename)
            print(f"Saved {name} to {model_filename}")
        
        # Save scaler
        scaler_filename = "/home/ubuntu/models/scaler.joblib"
        joblib.dump(self.scaler, scaler_filename)
        print(f"Saved scaler to {scaler_filename}")
        
        print("All models saved successfully!")
    
    def run_complete_analysis(self):
        """Run the complete fraud detection analysis pipeline."""
        print("Starting Complete Fraud Detection Analysis")
        print("=" * 50)
        
        # Load and explore data
        self.load_and_explore_data()
        
        # Create visualizations
        self.visualize_data_distribution()
        
        # Preprocess data
        self.preprocess_data()
        
        # Train models
        self.train_models()
        
        # Evaluate models
        self.evaluate_models()
        
        # Create performance visualizations
        self.create_performance_visualizations()
        
        # Generate report
        self.generate_detailed_report()
        
        # Save models
        self.save_models()
        
        print("\n" + "=" * 50)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("Generated files:")
        print("- data_distribution_analysis.png")
        print("- model_performance_analysis.png")
        print("- fraud_detection_report.txt")
        print("- models/ directory with saved models")
        print("=" * 50)

if __name__ == "__main__":
    # Initialize and run the fraud detection system
    fraud_detector = FraudDetectionSystem('/home/ubuntu/credit_card_fraud_dataset.csv')
    fraud_detector.run_complete_analysis()

