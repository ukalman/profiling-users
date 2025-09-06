#!/usr/bin/env python3
"""
Script to test Hugging Face space with social media data
Processes CSV file and classifies user profiles using the HF space API
"""

import pandas as pd
import numpy as np
from gradio_client import Client
import time
import logging
from typing import List, Dict, Any
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../hf_space_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HuggingFaceSpaceTester:
    def __init__(self, space_name: str = "CatoEr/Profiling-Social-Media-Users"):
        """Initialize the HF space client"""
        self.space_name = space_name
        self.client = None
        self.max_tweets_per_user = 20
        self.results = []
        
    def connect_to_space(self):
        """Connect to the Hugging Face space"""
        try:
            logger.info(f"Connecting to Hugging Face space: {self.space_name}")
            self.client = Client(self.space_name)
            logger.info("Successfully connected to HF space")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to HF space: {str(e)}")
            return False
    
    def load_csv_data(self, csv_path: str) -> pd.DataFrame:
        """Load and validate CSV data"""
        try:
            logger.info(f"Loading CSV data from: {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Validate required columns
            required_columns = ['post_owner.name', 'text']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Remove rows with empty text or post_owner.name
            initial_rows = len(df)
            df = df.dropna(subset=['post_owner.name', 'text'])
            df = df[df['text'].str.strip() != '']
            df = df[df['post_owner.name'].str.strip() != '']
            
            logger.info(f"Loaded {len(df)} valid rows (removed {initial_rows - len(df)} invalid rows)")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV data: {str(e)}")
            raise
    
    def group_texts_by_user(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Group texts by post_owner.name"""
        logger.info("Grouping texts by user...")
        
        user_texts = {}
        for _, row in df.iterrows():
            user_name = row['post_owner.name']
            text = str(row['text']).strip()
            
            if user_name not in user_texts:
                user_texts[user_name] = []
            
            # Limit to max_tweets_per_user
            if len(user_texts[user_name]) < self.max_tweets_per_user:
                user_texts[user_name].append(text)
        
        logger.info(f"Found {len(user_texts)} unique users")
        
        # Log distribution of text counts
        text_counts = [len(texts) for texts in user_texts.values()]
        logger.info(f"Text count distribution - Min: {min(text_counts)}, Max: {max(text_counts)}, Avg: {np.mean(text_counts):.2f}")
        
        return user_texts
    
    def prepare_api_params(self, texts: List[str]) -> Dict[str, str]:
        """Prepare parameters for the /predict API call"""
        params = {}
        
        # Fill up to 20 parameters
        for i in range(self.max_tweets_per_user):
            param_name = f"param_{i}"
            if i < len(texts):
                # Clean and truncate text if too long
                text = texts[i][:1000]  # Limit text length to avoid API issues
                params[param_name] = text
            else:
                # Fill empty slots with empty strings
                params[param_name] = ""
        
        return params
    
    def classify_user(self, user_name: str, texts: List[str]) -> Dict[str, Any]:
        """Classify a single user using the HF space API"""
        try:
            logger.info(f"Classifying user: {user_name} (with {len(texts)} texts)")
            
            # Prepare API parameters
            params = self.prepare_api_params(texts)
            
            # Call the API
            result = self.client.predict(
                **params,
                api_name="/predict"
            )
            
            # The API returns a string with the classification result
            classification = result if isinstance(result, str) else str(result)
            
            logger.info(f"Successfully classified user: {user_name}")
            
            return {
                'user_name': user_name,
                'num_texts': len(texts),
                'classification': classification,
                'status': 'success',
                'error': None,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error classifying user {user_name}: {str(e)}")
            return {
                'user_name': user_name,
                'num_texts': len(texts),
                'classification': None,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def process_all_users(self, user_texts: Dict[str, List[str]], delay_seconds: float = 1.0):
        """Process all users with rate limiting"""
        logger.info(f"Processing {len(user_texts)} users with {delay_seconds}s delay between requests")
        
        total_users = len(user_texts)
        processed = 0
        
        for user_name, texts in user_texts.items():
            processed += 1
            logger.info(f"Processing user {processed}/{total_users}: {user_name}")
            
            # Classify the user
            result = self.classify_user(user_name, texts)
            self.results.append(result)
            
            # Rate limiting
            if processed < total_users:  # Don't delay after the last request
                time.sleep(delay_seconds)
        
        logger.info(f"Completed processing all {total_users} users")
    
    def save_results(self, output_path: str):
        """Save results to CSV"""
        try:
            logger.info(f"Saving results to: {output_path}")
            
            if not self.results:
                logger.warning("No results to save")
                return
            
            # Create DataFrame from results
            results_df = pd.DataFrame(self.results)
            
            # Add summary statistics
            successful_classifications = len(results_df[results_df['status'] == 'success'])
            failed_classifications = len(results_df[results_df['status'] == 'error'])
            
            logger.info(f"Results summary - Successful: {successful_classifications}, Failed: {failed_classifications}")
            
            # Save to CSV
            results_df.to_csv(output_path, index=False)
            logger.info(f"Results saved successfully to {output_path}")
            
            return results_df
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise
    
    def run_full_pipeline(self, csv_path: str, output_path: str = None, delay_seconds: float = 1.0):
        """Run the complete pipeline"""
        try:
            # Set default output path if not provided
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"classification_results_{timestamp}.csv"
            
            logger.info("Starting HF Space testing pipeline")
            
            # Step 1: Connect to HF space
            if not self.connect_to_space():
                raise Exception("Failed to connect to Hugging Face space")
            
            # Step 2: Load CSV data
            df = self.load_csv_data(csv_path)
            
            # Step 3: Group texts by user
            user_texts = self.group_texts_by_user(df)
            
            # Step 4: Process all users
            self.process_all_users(user_texts, delay_seconds)
            
            # Step 5: Save results
            results_df = self.save_results(output_path)
            
            logger.info("Pipeline completed successfully!")
            return results_df
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

def main():
    """Main function to run the script"""
    # Configuration
    CSV_FILE_PATH = "../Islam 2024 Data.csv"  # Path to CSV file in parent directory
    OUTPUT_FILE_PATH = "../classification_results_islam_2024_01.csv"  # Save results in parent directory
    DELAY_BETWEEN_REQUESTS = 1.0  # seconds
    
    # Check if input file exists
    if not os.path.exists(CSV_FILE_PATH):
        logger.error(f"Input CSV file not found: {CSV_FILE_PATH}")
        return
    
    # Initialize and run the tester
    tester = HuggingFaceSpaceTester()
    
    try:
        results_df = tester.run_full_pipeline(
            csv_path=CSV_FILE_PATH,
            output_path=OUTPUT_FILE_PATH,
            delay_seconds=DELAY_BETWEEN_REQUESTS
        )
        
        # Print summary
        print("\n" + "="*50)
        print("CLASSIFICATION RESULTS SUMMARY")
        print("="*50)
        print(f"Total users processed: {len(results_df)}")
        print(f"Successful classifications: {len(results_df[results_df['status'] == 'success'])}")
        print(f"Failed classifications: {len(results_df[results_df['status'] == 'error'])}")
        print(f"Results saved to: {OUTPUT_FILE_PATH}")
        
        # Show sample results
        if len(results_df) > 0:
            print("\nSample results:")
            print(results_df[['user_name', 'num_texts', 'status', 'classification']].head())
        
    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
