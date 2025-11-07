#!/usr/bin/env python3
"""
Merge CSV files and create comprehensive analysis for research paper
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
import json

def extract_classification_data(classification_text):
    """Extract classification probabilities and predictions from the classification text"""
    if pd.isna(classification_text) or not classification_text:
        return {}
    
    data = {}
    
    # Extract each category section
    sections = classification_text.split('\n\n')
    
    for section in sections:
        if not section.strip():
            continue
            
        lines = section.strip().split('\n')
        if not lines:
            continue
            
        category = lines[0].strip()
        
        # Skip if this is not a valid category
        if category not in ['RACE', 'AGE', 'EDUCATION', 'GENDER', 'SEXUAL ORIENTATION']:
            continue
        
        probabilities = {}
        predicted = None
        
        for line in lines[1:]:
            line = line.strip()
            if line.startswith('Probabilities:'):
                continue
            elif '=' in line and '%' in line:
                # Parse probability line like "Asian = 84.22%"
                parts = line.split('=')
                if len(parts) == 2:
                    class_name = parts[0].strip()
                    prob_str = parts[1].strip().replace('%', '')
                    try:
                        probabilities[class_name] = float(prob_str)
                    except ValueError:
                        pass
            elif line.startswith('Predicted as:'):
                predicted = line.replace('Predicted as:', '').strip()
        
        data[category.lower().replace(' ', '_')] = {
            'probabilities': probabilities,
            'predicted': predicted
        }
    
    return data

def load_and_process_csv(filepath):
    """Load and process a CSV file"""
    print(f"Loading {filepath}...")
    
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows from {filepath}")
        
        # Extract classification data
        print("Extracting classification data...")
        classification_data = []
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"Processed {idx} rows...")
            
            extracted = extract_classification_data(row.get('classification', ''))
            classification_data.append(extracted)
        
        # Add extracted data to dataframe
        for category in ['race', 'age', 'education', 'gender', 'sexual_orientation']:
            df[f'{category}_predicted'] = [data.get(category, {}).get('predicted', None) for data in classification_data]
            
            # Add probability columns for each class
            if classification_data:
                all_classes = set()
                for data in classification_data:
                    if category in data and 'probabilities' in data[category]:
                        all_classes.update(data[category]['probabilities'].keys())
                
                for class_name in all_classes:
                    col_name = f'{category}_{class_name.lower().replace(" ", "_").replace("-", "_")}_prob'
                    df[col_name] = [
                        data.get(category, {}).get('probabilities', {}).get(class_name, 0) 
                        for data in classification_data
                    ]
        
        return df
        
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def merge_csvs(file1_path, file2_path, output_path):
    """Merge two CSV files"""
    print("Starting merge process...")
    
    df1 = load_and_process_csv(file1_path)
    df2 = load_and_process_csv(file2_path)
    
    if df1 is None or df2 is None:
        print("Error: Could not load one or both CSV files")
        return None
    
    print(f"File 1 shape: {df1.shape}")
    print(f"File 2 shape: {df2.shape}")
    
    # Merge dataframes
    merged_df = pd.concat([df1, df2], ignore_index=True)
    print(f"Merged shape: {merged_df.shape}")
    
    # Remove duplicates based on user_id if it exists
    if 'user_id' in merged_df.columns:
        initial_count = len(merged_df)
        merged_df = merged_df.drop_duplicates(subset=['user_id'], keep='first')
        print(f"Removed {initial_count - len(merged_df)} duplicates")
    
    # Save merged file
    print(f"Saving merged file to {output_path}...")
    merged_df.to_csv(output_path, index=False)
    
    return merged_df

def create_comprehensive_analysis(df, output_path):
    """Create comprehensive analysis report"""
    print("Creating comprehensive analysis...")
    
    analysis = {
        'metadata': {
            'total_users': len(df),
            'total_posts': df['num_posts'].sum() if 'num_posts' in df.columns else 0,
            'analysis_date': datetime.now().isoformat(),
            'description': 'Analysis of Facebook posts about Islam - User demographic and sentiment distribution'
        },
        'demographics': {},
        'sentiment_analysis': {},
        'cross_analysis': {}
    }
    
    # Demographic analysis
    categories = ['race', 'age', 'education', 'gender', 'sexual_orientation']
    
    for category in categories:
        pred_col = f'{category}_predicted'
        if pred_col in df.columns:
            # Overall distribution
            distribution = df[pred_col].value_counts()
            percentages = df[pred_col].value_counts(normalize=True) * 100
            
            analysis['demographics'][category] = {
                'distribution': distribution.to_dict(),
                'percentages': {k: round(v, 2) for k, v in percentages.to_dict().items()},
                'total_classified': distribution.sum(),
                'unclassified': len(df) - distribution.sum()
            }
            
            print(f"{category.upper()} Distribution:")
            for class_name, count in distribution.items():
                percentage = percentages[class_name]
                print(f"  {class_name}: {count} users ({percentage:.2f}%)")
    
    # Sentiment analysis
    if 'voted_sentiment' in df.columns:
        sentiment_dist = df['voted_sentiment'].value_counts()
        sentiment_pct = df['voted_sentiment'].value_counts(normalize=True) * 100
        
        analysis['sentiment_analysis'] = {
            'overall_distribution': sentiment_dist.to_dict(),
            'overall_percentages': {k: round(v, 2) for k, v in sentiment_pct.to_dict().items()},
            'by_category': {}
        }
        
        print(f"\nOVERALL SENTIMENT Distribution:")
        for sentiment, count in sentiment_dist.items():
            percentage = sentiment_pct[sentiment]
            print(f"  {sentiment}: {count} users ({percentage:.2f}%)")
        
        # Sentiment by demographic category
        for category in categories:
            pred_col = f'{category}_predicted'
            if pred_col in df.columns:
                sentiment_by_cat = pd.crosstab(df[pred_col], df['voted_sentiment'])
                sentiment_by_cat_pct = pd.crosstab(df[pred_col], df['voted_sentiment'], normalize='index') * 100
                
                analysis['sentiment_analysis']['by_category'][category] = {}
                
                print(f"\nSentiment by {category.upper()}:")
                for class_name in sentiment_by_cat.index:
                    class_sentiments = {}
                    class_sentiments_pct = {}
                    
                    print(f"  {class_name}:")
                    for sentiment in sentiment_by_cat.columns:
                        count = sentiment_by_cat.loc[class_name, sentiment]
                        pct = sentiment_by_cat_pct.loc[class_name, sentiment]
                        class_sentiments[sentiment] = count
                        class_sentiments_pct[sentiment] = round(pct, 2)
                        print(f"    {sentiment}: {count} ({pct:.2f}%)")
                    
                    analysis['sentiment_analysis']['by_category'][category][class_name] = {
                        'counts': class_sentiments,
                        'percentages': class_sentiments_pct
                    }
    
    # Cross-category analysis
    print(f"\nCROSS-CATEGORY ANALYSIS:")
    
    # Age vs Gender
    if 'age_predicted' in df.columns and 'gender_predicted' in df.columns:
        age_gender = pd.crosstab(df['age_predicted'], df['gender_predicted'])
        analysis['cross_analysis']['age_vs_gender'] = age_gender.to_dict()
        print("Age vs Gender distribution:")
        print(age_gender)
    
    # Race vs Education
    if 'race_predicted' in df.columns and 'education_predicted' in df.columns:
        race_education = pd.crosstab(df['race_predicted'], df['education_predicted'])
        analysis['cross_analysis']['race_vs_education'] = race_education.to_dict()
        print("\nRace vs Education distribution:")
        print(race_education)
    
    # Additional statistics
    if 'num_posts' in df.columns:
        analysis['post_statistics'] = {
            'total_posts': int(df['num_posts'].sum()),
            'avg_posts_per_user': round(df['num_posts'].mean(), 2),
            'median_posts_per_user': int(df['num_posts'].median()),
            'max_posts_per_user': int(df['num_posts'].max()),
            'min_posts_per_user': int(df['num_posts'].min())
        }
        
        print(f"\nPOST STATISTICS:")
        print(f"Total posts analyzed: {analysis['post_statistics']['total_posts']:,}")
        print(f"Average posts per user: {analysis['post_statistics']['avg_posts_per_user']}")
        print(f"Median posts per user: {analysis['post_statistics']['median_posts_per_user']}")
    
    # Save analysis
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    return analysis

def create_research_summary(analysis, output_path):
    """Create a research-friendly summary document"""
    
    summary_text = f"""
# Facebook Posts about Islam - Demographic and Sentiment Analysis Report

**Analysis Date:** {analysis['metadata']['analysis_date']}
**Total Users Analyzed:** {analysis['metadata']['total_users']:,}
**Total Posts Analyzed:** {analysis['metadata']['total_posts']:,}

## Executive Summary

This analysis examines the demographic distribution and sentiment patterns of {analysis['metadata']['total_users']:,} Facebook users who posted about Islam, covering {analysis['metadata']['total_posts']:,} individual posts.

## Demographic Distribution

### Race/Ethnicity Distribution
"""
    
    if 'race' in analysis['demographics']:
        for race, percentage in analysis['demographics']['race']['percentages'].items():
            if race and race != 'None':
                summary_text += f"- **{race}:** {percentage}%\n"
    
    summary_text += "\n### Age Distribution\n"
    if 'age' in analysis['demographics']:
        for age, percentage in analysis['demographics']['age']['percentages'].items():
            if age and age != 'None':
                summary_text += f"- **{age}:** {percentage}%\n"
    
    summary_text += "\n### Education Level Distribution\n"
    if 'education' in analysis['demographics']:
        for edu, percentage in analysis['demographics']['education']['percentages'].items():
            if edu and edu != 'None':
                summary_text += f"- **{edu}:** {percentage}%\n"
    
    summary_text += "\n### Gender Distribution\n"
    if 'gender' in analysis['demographics']:
        for gender, percentage in analysis['demographics']['gender']['percentages'].items():
            if gender and gender != 'None':
                summary_text += f"- **{gender}:** {percentage}%\n"
    
    summary_text += "\n### Sexual Orientation Distribution\n"
    if 'sexual_orientation' in analysis['demographics']:
        for orientation, percentage in analysis['demographics']['sexual_orientation']['percentages'].items():
            if orientation and orientation != 'None':
                summary_text += f"- **{orientation}:** {percentage}%\n"
    
    summary_text += "\n## Sentiment Analysis\n\n### Overall Sentiment Distribution\n"
    if 'sentiment_analysis' in analysis:
        for sentiment, percentage in analysis['sentiment_analysis']['overall_percentages'].items():
            summary_text += f"- **{sentiment.capitalize()}:** {percentage}%\n"
    
    summary_text += "\n### Sentiment Distribution by Demographics\n"
    
    if 'sentiment_analysis' in analysis and 'by_category' in analysis['sentiment_analysis']:
        for category, category_data in analysis['sentiment_analysis']['by_category'].items():
            if category_data:
                summary_text += f"\n#### Sentiment by {category.replace('_', ' ').title()}\n"
                for class_name, sentiments in category_data.items():
                    if class_name and class_name != 'None':
                        summary_text += f"\n**{class_name}:**\n"
                        for sentiment, pct in sentiments['percentages'].items():
                            summary_text += f"- {sentiment.capitalize()}: {pct}%\n"
    
    if 'post_statistics' in analysis:
        summary_text += f"""
## Post Activity Statistics

- **Total posts analyzed:** {analysis['post_statistics']['total_posts']:,}
- **Average posts per user:** {analysis['post_statistics']['avg_posts_per_user']}
- **Median posts per user:** {analysis['post_statistics']['median_posts_per_user']}
- **Most active user:** {analysis['post_statistics']['max_posts_per_user']} posts

## Research Implications

This analysis reveals significant patterns in how different demographic groups engage with Islamic content on Facebook. The data provides insights into:

1. **Demographic Representation:** The distribution shows which population segments are most actively discussing Islamic topics
2. **Sentiment Patterns:** Different demographic groups show varying sentiment patterns in their posts about Islam
3. **Engagement Levels:** Post frequency varies significantly across different user segments

## Methodology

- **Data Source:** Facebook posts containing Islamic content
- **Classification Method:** Automated demographic and sentiment classification using AI models
- **Sample Size:** {analysis['metadata']['total_users']:,} unique users, {analysis['metadata']['total_posts']:,} posts
- **Analysis Categories:** Race/Ethnicity, Age, Education, Gender, Sexual Orientation, Sentiment

---
*This report was generated automatically from the merged dataset analysis.*
"""
    
    with open(output_path, 'w') as f:
        f.write(summary_text)
    
    print(f"Research summary saved to {output_path}")

def main():
    """Main execution function"""
    
    # File paths
    file1 = "/Users/baranorhan/Documents/GitHub/profiling-users/HF Space API/single_space_results_updated.csv"
    file2 = "/Users/baranorhan/Documents/GitHub/profiling-users/HF Space API/Facebook Results.csv"
    merged_output = "/Users/baranorhan/Documents/GitHub/profiling-users/HF Space API/merged_facebook_islam_analysis.csv"
    analysis_output = "/Users/baranorhan/Documents/GitHub/profiling-users/HF Space API/comprehensive_analysis.json"
    summary_output = "/Users/baranorhan/Documents/GitHub/profiling-users/HF Space API/research_summary_report.md"
    
    print("=== Facebook Islam Posts Analysis ===")
    print("Merging CSV files and creating comprehensive analysis...\n")
    
    # Merge CSV files
    merged_df = merge_csvs(file1, file2, merged_output)
    
    if merged_df is not None:
        print(f"\nMerge completed successfully!")
        print(f"Merged file saved to: {merged_output}")
        
        # Create comprehensive analysis
        analysis = create_comprehensive_analysis(merged_df, analysis_output)
        print(f"Detailed analysis saved to: {analysis_output}")
        
        # Create research summary
        create_research_summary(analysis, summary_output)
        
        print(f"\n=== Analysis Complete ===")
        print(f"Files created:")
        print(f"1. Merged dataset: {merged_output}")
        print(f"2. Detailed analysis: {analysis_output}")
        print(f"3. Research summary: {summary_output}")
    else:
        print("Merge failed!")

if __name__ == "__main__":
    main()