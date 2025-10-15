#!/usr/bin/env python3
"""
Large Dataset Processor for Social Media Profiling
Processes 21 CSV files (~2GB) with dual-space HF classification
"""

import pandas as pd
import numpy as np
import logging
import time
import os
import glob
from pathlib import Path
from gradio_client import Client
import concurrent.futures
from threading import Lock
import argparse
from datetime import datetime

# Configuration
LARGE_DATASET_PATH = "/Users/baranorhan/Downloads/ProfilingSocialMedia/"
SENTIMENT_FILE = "user_sentiment_from_blocks.csv"
SPACE_1 = "CatoEr/Profiling-Social-Media-Users"
SPACE_2 = "baranorhan/Profiling-Social-Media-Users"
BATCH_SIZE = 1000  # Process users in batches
MAX_WORKERS = 2    # One per space
CHUNK_SIZE = 50000 # Read CSV files in chunks to manage memory

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('large_dataset_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LargeDatasetProcessor:
    def __init__(self):
        self.client_1 = None
        self.client_2 = None
        self.results = []
        self.processed_count = 0
        self.failed_count = 0
        self.lock = Lock()
        self.space_usage = {SPACE_1: 0, SPACE_2: 0}
        
    def connect_to_spaces(self):
        """Connect to both HuggingFace spaces"""
        logger.info(f"Connecting to HF Space 1: {SPACE_1}")
        try:
            self.client_1 = Client(SPACE_1)
            logger.info("Successfully connected to HF Space 1")
        except Exception as e:
            logger.error(f"Failed to connect to Space 1: {e}")
            raise
            
        logger.info(f"Connecting to HF Space 2: {SPACE_2}")
        try:
            self.client_2 = Client(SPACE_2)
            logger.info("Successfully connected to HF Space 2")
        except Exception as e:
            logger.error(f"Failed to connect to Space 2: {e}")
            raise
    
    def load_csv_files_in_chunks(self):
        """Load all CSV files efficiently using chunking"""
        logger.info("Loading large dataset files...")
        csv_files = glob.glob(os.path.join(LARGE_DATASET_PATH, "*.csv"))
        csv_files.sort()  # Process in chronological order
        
        logger.info(f"Found {len(csv_files)} CSV files to process")
        
        all_posts = []
        total_rows = 0
        
        for i, file_path in enumerate(csv_files, 1):
            file_name = os.path.basename(file_path)
            logger.info(f"Processing file {i}/{len(csv_files)}: {file_name}")
            
            try:
                # Read file in chunks to manage memory
                chunk_count = 0
                file_rows = 0
                
                for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE):
                    chunk_count += 1
                    chunk_rows = len(chunk)
                    file_rows += chunk_rows
                    
                    # Filter required columns and clean data
                    required_cols = ['post_owner.id', 'surface.id', 'text', 'post_owner.name']
                    available_cols = [col for col in required_cols if col in chunk.columns]
                    
                    if 'post_owner.id' not in available_cols or 'text' not in available_cols:
                        logger.warning(f"Skipping chunk from {file_name} - missing required columns")
                        continue
                    
                    chunk_filtered = chunk[available_cols].copy()
                    
                    # Clean data
                    chunk_filtered = chunk_filtered.dropna(subset=['post_owner.id', 'text'])
                    chunk_filtered = chunk_filtered[chunk_filtered['text'].str.strip() != '']
                    
                    # Add to collection
                    all_posts.append(chunk_filtered)
                    
                    if chunk_count % 10 == 0:
                        logger.info(f"  Processed {chunk_count} chunks from {file_name}")
                
                total_rows += file_rows
                logger.info(f"  Loaded {file_rows:,} rows from {file_name}")
                
            except Exception as e:
                logger.error(f"Error processing {file_name}: {e}")
                continue
        
        if not all_posts:
            raise ValueError("No valid data found in CSV files")
        
        # Combine all chunks
        logger.info("Combining all data chunks...")
        combined_df = pd.concat(all_posts, ignore_index=True)
        logger.info(f"Total posts loaded: {len(combined_df):,} from {total_rows:,} original rows")
        
        return combined_df
    
    def load_sentiment_data(self):
        """Load sentiment analysis results"""
        logger.info("Loading sentiment analysis results...")
        sentiment_df = pd.read_csv(SENTIMENT_FILE)
        logger.info(f"Loaded {len(sentiment_df):,} sentiment records")
        return sentiment_df
    
    def merge_and_vote(self, posts_df, sentiment_df):
        """Merge posts with sentiment and apply voting mechanism"""
        logger.info("Merging posts with sentiment data...")
        
        # Merge on surface.id
        merged_df = posts_df.merge(
            sentiment_df, 
            left_on='surface.id', 
            right_on='surface.id', 
            how='inner'
        )
        
        logger.info(f"Successfully merged: {len(merged_df):,} records with both posts and sentiment")
        
        # Clean merged data
        merged_df = merged_df.dropna(subset=['post_owner.id', 'text', 'sentiment_overall'])
        merged_df = merged_df[merged_df['text'].str.strip() != '']
        
        logger.info(f"After cleaning: {len(merged_df):,} valid rows")
        
        # Apply voting mechanism
        logger.info("Applying voting mechanism...")
        
        def vote_sentiment(group):
            sentiments = group['sentiment_overall'].tolist()
            sentiment_counts = pd.Series(sentiments).value_counts()
            
            # Get most common sentiment
            winning_sentiment = sentiment_counts.index[0]
            winning_count = sentiment_counts.iloc[0]
            total_count = len(sentiments)
            
            # Calculate confidence
            confidence = winning_count / total_count
            winning_percentage = confidence * 100
            is_unanimous = len(sentiment_counts) == 1
            
            # Sample surface IDs and sentiments for reference
            surface_ids_sample = group['surface.id'].head(3).tolist()
            individual_sentiments_sample = group['sentiment_overall'].head(3).tolist()
            
            return pd.Series({
                'user_id': group['post_owner.id'].iloc[0],
                'user_name': group['post_owner.name'].iloc[0] if 'post_owner.name' in group.columns else 'Unknown',
                'num_posts': len(group),
                'voted_sentiment': winning_sentiment,
                'sentiment_confidence': confidence,
                'winning_percentage': winning_percentage,
                'is_unanimous': is_unanimous,
                'surface_ids_sample': str(surface_ids_sample),
                'individual_sentiments_sample': str(individual_sentiments_sample),
                'sentiment_pos': (pd.Series(sentiments) == 'pos').sum(),
                'sentiment_neg': (pd.Series(sentiments) == 'neg').sum(),
                'first_post_sample': group['text'].iloc[0][:200] + '...' if len(group['text'].iloc[0]) > 200 else group['text'].iloc[0]
            })
        
        voted_df = merged_df.groupby('post_owner.id').apply(vote_sentiment).reset_index(drop=True)
        
        logger.info(f"Voting completed. Found {len(voted_df):,} unique users")
        
        # Voting quality analysis
        unanimous_count = voted_df['is_unanimous'].sum()
        high_conf_count = len(voted_df[voted_df['sentiment_confidence'] >= 0.7])
        
        logger.info(f"Voting quality:")
        logger.info(f"  - High confidence (≥70%): {high_conf_count}/{len(voted_df)} ({high_conf_count/len(voted_df)*100:.1f}%)")
        logger.info(f"  - Unanimous sentiment: {unanimous_count}/{len(voted_df)} ({unanimous_count/len(voted_df)*100:.1f}%)")
        
        return voted_df
    
    def classify_user(self, user_data, space_name, client):
        """Classify a single user using specified space"""
        try:
            user_id = user_data['user_id']
            user_name = user_data['user_name']
            voted_sentiment = user_data['voted_sentiment']
            num_posts = user_data['num_posts']
            confidence = user_data['sentiment_confidence']
            first_post = user_data['first_post_sample']
            
            logger.info(f"[{space_name}] Classifying user {user_id} ({user_name}): {num_posts} posts, sentiment: {voted_sentiment} ({confidence:.2f} confidence)")
            
            # Format post with sentiment context
            enhanced_post = f"[POST_SENTIMENT: {voted_sentiment}] [USER_SENTIMENT: {voted_sentiment}] {first_post}"
            
            # Prepare parameters for HF Space
            params = {f'param_{i}': '' for i in range(20)}
            params['param_0'] = enhanced_post
            
            # Call HF Space
            result = client.predict(**params, api_name='/predict')
            
            with self.lock:
                self.processed_count += 1
                self.space_usage[space_name] += 1
                
                # Create result record
                result_record = user_data.copy()
                result_record['classification'] = result
                result_record['status'] = 'success'
                result_record['error'] = None
                result_record['processed_by'] = space_name
                result_record['timestamp'] = datetime.now().isoformat()
                
                self.results.append(result_record)
                
                logger.info(f"[{space_name}] Successfully classified user {user_id}")
                logger.info(f"Progress: {self.processed_count}/{len(self.user_queue)} users completed ({self.processed_count/len(self.user_queue)*100:.1f}%)")
            
        except Exception as e:
            with self.lock:
                self.failed_count += 1
                
                # Create failure record
                result_record = user_data.copy()
                result_record['classification'] = None
                result_record['status'] = 'failed'
                result_record['error'] = str(e)
                result_record['processed_by'] = space_name
                result_record['timestamp'] = datetime.now().isoformat()
                
                self.results.append(result_record)
                
                logger.error(f"[{space_name}] Failed to classify user {user_data['user_id']}: {e}")
    
    def process_batch(self, batch_users):
        """Process a batch of users using both spaces in parallel"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            
            for i, user_data in enumerate(batch_users):
                # Alternate between spaces for load balancing
                if i % 2 == 0:
                    client = self.client_1
                    space_name = SPACE_1
                else:
                    client = self.client_2
                    space_name = SPACE_2
                
                future = executor.submit(self.classify_user, user_data, space_name, client)
                futures.append(future)
            
            # Wait for all tasks to complete
            concurrent.futures.wait(futures)
    
    def process_users(self, voted_df):
        """Process all users in batches"""
        self.user_queue = voted_df.to_dict('records')
        total_users = len(self.user_queue)
        
        logger.info(f"Processing {total_users:,} users in batches of {BATCH_SIZE}")
        
        start_time = time.time()
        
        # Process in batches to manage memory and provide progress updates
        for batch_start in range(0, total_users, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_users)
            batch_users = self.user_queue[batch_start:batch_end]
            
            batch_num = (batch_start // BATCH_SIZE) + 1
            total_batches = (total_users + BATCH_SIZE - 1) // BATCH_SIZE
            
            logger.info(f"\n🔄 Processing batch {batch_num}/{total_batches} (users {batch_start + 1}-{batch_end})")
            
            batch_start_time = time.time()
            self.process_batch(batch_users)
            batch_time = time.time() - batch_start_time
            
            logger.info(f"✅ Batch {batch_num} completed in {batch_time:.1f} seconds")
            
            # Progress update
            elapsed_time = time.time() - start_time
            avg_time_per_user = elapsed_time / max(self.processed_count, 1)
            remaining_users = total_users - self.processed_count
            estimated_remaining_time = remaining_users * avg_time_per_user
            
            logger.info(f"📊 Progress: {self.processed_count}/{total_users} users ({self.processed_count/total_users*100:.1f}%)")
            logger.info(f"⏱️  Average time per user: {avg_time_per_user:.2f}s")
            logger.info(f"🕐 Estimated remaining time: {estimated_remaining_time/60:.1f} minutes")
        
        total_time = time.time() - start_time
        logger.info(f"\n🎉 All batches completed in {total_time/60:.1f} minutes")
        
        return total_time
    
    def save_results(self, output_file="large_dataset_classification_results.csv"):
        """Save results to CSV"""
        if not self.results:
            logger.warning("No results to save")
            return
        
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to: {output_file}")
        
        return output_file
    
    def run(self, sample_size=None):
        """Main processing pipeline"""
        logger.info("Starting large dataset processing pipeline")
        logger.info("="*60)
        
        start_time = time.time()
        
        try:
            # Connect to spaces
            self.connect_to_spaces()
            
            # Load data
            posts_df = self.load_csv_files_in_chunks()
            sentiment_df = self.load_sentiment_data()
            
            # Merge and vote
            voted_df = self.merge_and_vote(posts_df, sentiment_df)
            
            # Optional sampling for testing
            if sample_size:
                logger.info(f"Sampling {sample_size} users for testing")
                voted_df = voted_df.head(sample_size)
            
            # Process users
            processing_time = self.process_users(voted_df)
            
            # Save results
            output_file = self.save_results()
            
            # Summary
            total_time = time.time() - start_time
            
            logger.info("\n" + "="*70)
            logger.info("LARGE DATASET PROCESSING SUMMARY")
            logger.info("="*70)
            logger.info(f"Total users processed: {len(voted_df):,}")
            logger.info(f"Successful classifications: {self.processed_count}")
            logger.info(f"Failed classifications: {self.failed_count}")
            logger.info(f"Success rate: {self.processed_count/(self.processed_count + self.failed_count)*100:.1f}%")
            logger.info(f"Total processing time: {total_time/60:.1f} minutes")
            logger.info(f"Average time per user: {processing_time/max(self.processed_count, 1):.2f} seconds")
            logger.info(f"Results saved to: {output_file}")
            logger.info(f"Space utilization: {self.space_usage}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

def main():
    global BATCH_SIZE
    
    parser = argparse.ArgumentParser(description='Process large social media dataset')
    parser.add_argument('--sample', type=int, help='Sample size for testing (default: process all)')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help=f'Batch size (default: {BATCH_SIZE})')
    
    args = parser.parse_args()
    
    if args.batch_size:
        BATCH_SIZE = args.batch_size
    
    processor = LargeDatasetProcessor()
    processor.run(sample_size=args.sample)

if __name__ == "__main__":
    main()