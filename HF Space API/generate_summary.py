#!/usr/bin/env python3
"""
Script to generate summary statistics from classification results
Parses the classification results CSV and creates aggregated statistics
"""

import pandas as pd
import re
from typing import Dict, List, Tuple
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ClassificationSummarizer:
    def __init__(self):
        """Initialize the summarizer"""
        self.categories = ['RACE', 'AGE', 'EDUCATION', 'GENDER', 'SEXUAL ORIENTATION']
        self.summary_data = {}
        
    def parse_classification_text(self, classification_text: str) -> Dict[str, Dict[str, float]]:
        """Parse the classification text to extract probabilities and predictions"""
        if pd.isna(classification_text) or classification_text is None:
            return {}
        
        results = {}
        
        # Split by categories - improved pattern to capture each section properly
        for category in self.categories:
            # Pattern to match the entire category section
            category_pattern = rf"{category}\s*\n\s*Probabilities:(.*?)Predicted as:\s*([A-Za-z\s\-]+?)(?=\n|$)"
            category_match = re.search(category_pattern, classification_text, re.DOTALL | re.IGNORECASE)
            
            if category_match:
                probabilities_text = category_match.group(1)
                prediction = category_match.group(2).strip()
                
                # Clean up prediction - remove any trailing newlines or extra text
                prediction = re.sub(r'\n.*$', '', prediction).strip()
                
                # Extract probabilities
                prob_pattern = r"([A-Za-z\s\-]+)\s*=\s*([\d.]+)%"
                probabilities = {}
                
                for prob_match in re.finditer(prob_pattern, probabilities_text):
                    class_name = prob_match.group(1).strip()
                    probability = float(prob_match.group(2))
                    probabilities[class_name] = probability
                
                results[category] = {
                    'probabilities': probabilities,
                    'prediction': prediction
                }
        
        return results
    
    def process_results_csv(self, csv_path: str) -> pd.DataFrame:
        """Process the classification results CSV"""
        logger.info(f"Loading classification results from: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            
            # Filter successful classifications only
            successful_df = df[df['status'] == 'success'].copy()
            logger.info(f"Processing {len(successful_df)} successful classifications out of {len(df)} total")
            
            # Parse each classification
            parsed_results = []
            
            for idx, row in successful_df.iterrows():
                user_name = row['user_name']
                classification_text = row['classification']
                
                logger.info(f"Parsing classification for user: {user_name}")
                
                parsed = self.parse_classification_text(classification_text)
                
                if parsed:
                    result_row = {'user_name': user_name}
                    
                    # Add predictions and probabilities for each category
                    for category, data in parsed.items():
                        # Add prediction
                        result_row[f'{category}_prediction'] = data.get('prediction', 'Unknown')
                        
                        # Add probabilities
                        for class_name, prob in data.get('probabilities', {}).items():
                            result_row[f'{category}_{class_name}_probability'] = prob
                    
                    parsed_results.append(result_row)
            
            parsed_df = pd.DataFrame(parsed_results)
            logger.info(f"Successfully parsed {len(parsed_df)} classifications")
            
            return parsed_df
            
        except Exception as e:
            logger.error(f"Error processing results CSV: {str(e)}")
            raise
    
    def generate_summary_statistics(self, parsed_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Generate summary statistics for each category"""
        logger.info("Generating summary statistics...")
        
        summaries = {}
        
        for category in self.categories:
            prediction_col = f'{category}_prediction'
            
            if prediction_col in parsed_df.columns:
                # Count predictions
                prediction_counts = parsed_df[prediction_col].value_counts()
                prediction_percentages = (prediction_counts / len(parsed_df) * 100).round(2)
                
                # Create summary DataFrame
                summary_df = pd.DataFrame({
                    'Category': category,
                    'Class': prediction_counts.index,
                    'Count': prediction_counts.values,
                    'Percentage': prediction_percentages.values
                })
                
                # Calculate average probabilities for each class
                prob_columns = [col for col in parsed_df.columns if col.startswith(f'{category}_') and col.endswith('_probability')]
                
                avg_probabilities = []
                for class_name in prediction_counts.index:
                    # Find the probability column for this class
                    prob_col = f'{category}_{class_name}_probability'
                    if prob_col in parsed_df.columns:
                        avg_prob = parsed_df[prob_col].mean()
                        avg_probabilities.append(round(avg_prob, 2))
                    else:
                        avg_probabilities.append(0.0)
                
                summary_df['Average_Probability'] = avg_probabilities
                summaries[category] = summary_df
                
                logger.info(f"Summary for {category}:")
                logger.info(f"\n{summary_df.to_string(index=False)}")
        
        return summaries
    
    def create_overall_summary(self, summaries: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create an overall summary combining all categories"""
        logger.info("Creating overall summary...")
        
        # Combine all summaries
        all_summaries = []
        for category, summary_df in summaries.items():
            all_summaries.append(summary_df)
        
        overall_summary = pd.concat(all_summaries, ignore_index=True)
        
        return overall_summary
    
    def save_summaries(self, summaries: Dict[str, pd.DataFrame], overall_summary: pd.DataFrame, 
                      output_dir: str = ".", base_filename: str = "classification_summary"):
        """Save all summaries to CSV files"""
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save overall summary
        overall_path = os.path.join(output_dir, f"{base_filename}_overall_{timestamp}.csv")
        overall_summary.to_csv(overall_path, index=False)
        logger.info(f"Overall summary saved to: {overall_path}")
        
        # Save individual category summaries
        for category, summary_df in summaries.items():
            category_clean = category.lower().replace(' ', '_')
            category_path = os.path.join(output_dir, f"{base_filename}_{category_clean}_{timestamp}.csv")
            summary_df.to_csv(category_path, index=False)
            logger.info(f"{category} summary saved to: {category_path}")
        
        return overall_path, {category: f"{base_filename}_{category.lower().replace(' ', '_')}_{timestamp}.csv" 
                             for category in summaries.keys()}
    
    def generate_detailed_breakdown(self, parsed_df: pd.DataFrame) -> pd.DataFrame:
        """Generate a detailed breakdown showing user-by-user predictions"""
        logger.info("Generating detailed user breakdown...")
        
        # Select only prediction columns
        prediction_columns = ['user_name'] + [col for col in parsed_df.columns if col.endswith('_prediction')]
        
        if len(prediction_columns) > 1:  # More than just user_name
            detailed_df = parsed_df[prediction_columns].copy()
            
            # Rename columns for better readability
            column_mapping = {'user_name': 'User_Name'}
            for col in prediction_columns[1:]:  # Skip user_name
                category = col.replace('_prediction', '').replace('_', ' ').title()
                column_mapping[col] = f'{category}_Prediction'
            
            detailed_df = detailed_df.rename(columns=column_mapping)
            
            return detailed_df
        else:
            return pd.DataFrame()
    
    def run_full_analysis(self, results_csv_path: str, output_dir: str = "."):
        """Run the complete analysis pipeline"""
        try:
            logger.info("Starting classification summary analysis...")
            
            # Step 1: Process results CSV
            parsed_df = self.process_results_csv(results_csv_path)
            
            if parsed_df.empty:
                logger.warning("No data to analyze")
                return
            
            # Step 2: Generate summary statistics
            summaries = self.generate_summary_statistics(parsed_df)
            
            # Step 3: Create overall summary
            overall_summary = self.create_overall_summary(summaries)
            
            # Step 4: Generate detailed breakdown
            detailed_breakdown = self.generate_detailed_breakdown(parsed_df)
            
            # Step 5: Save all results
            overall_path, category_paths = self.save_summaries(summaries, overall_summary, output_dir)
            
            # Save detailed breakdown
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            detailed_path = os.path.join(output_dir, f"classification_detailed_breakdown_{timestamp}.csv")
            detailed_breakdown.to_csv(detailed_path, index=False)
            logger.info(f"Detailed breakdown saved to: {detailed_path}")
            
            # Print summary to console
            print("\n" + "="*60)
            print("CLASSIFICATION SUMMARY ANALYSIS")
            print("="*60)
            print(f"Total users analyzed: {len(parsed_df)}")
            print(f"Categories analyzed: {len(summaries)}")
            print("\nOverall Summary:")
            print(overall_summary.to_string(index=False))
            
            print(f"\nFiles generated:")
            print(f"- Overall summary: {overall_path}")
            print(f"- Detailed breakdown: {detailed_path}")
            for category, path in category_paths.items():
                print(f"- {category} summary: {path}")
            
            logger.info("Analysis completed successfully!")
            
            return {
                'parsed_data': parsed_df,
                'summaries': summaries,
                'overall_summary': overall_summary,
                'detailed_breakdown': detailed_breakdown
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise

def main():
    """Main function to run the summary analysis"""
    
    # Configuration
    RESULTS_CSV_PATH = "../classification_results.csv"  # Path to your classification results in parent directory
    OUTPUT_DIR = ".."  # Directory to save summary files in parent directory
    
    # Check if results file exists
    if not os.path.exists(RESULTS_CSV_PATH):
        logger.error(f"Results CSV file not found: {RESULTS_CSV_PATH}")
        print(f"Error: Results file '{RESULTS_CSV_PATH}' not found.")
        print("Please make sure you have run the classification script first.")
        return
    
    # Initialize and run the summarizer
    summarizer = ClassificationSummarizer()
    
    try:
        results = summarizer.run_full_analysis(RESULTS_CSV_PATH, OUTPUT_DIR)
        print("\nSummary analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Summary analysis failed: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
