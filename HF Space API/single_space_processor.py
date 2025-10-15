#!/usr/bin/env python3
"""
Single Space Incremental Processor
Uses only baranorhan/Profiling-Social-Media-Users space
"""

import pandas as pd
import numpy as np
import logging
import time
import os
import glob
import json
from pathlib import Path
from gradio_client import Client
import argparse
from datetime import datetime

# Configuration - UPDATE THESE PATHS FOR YOUR SYSTEM
LARGE_DATASET_PATH = "/Users/baranorhan/Downloads/ProfilingSocialMedia/"  # UPDATE: Path to your 21 CSV files
SENTIMENT_FILE = "user_sentiment_from_blocks.csv"  # This file should be in the same folder as this script
SPACE = "baranorhan/Profiling-Social-Media-Users"  # UPDATE: Change to "CatoEr/Profiling-Social-Media-Users" for team member 2
BATCH_SIZE = 100  # Smaller batches for single space
CHUNK_SIZE = 50000

# Incremental processing files
PROGRESS_FILE = "single_space_progress.json"
PROCESSED_USERS_FILE = "single_space_processed_users.json"
RESULTS_FILE = "single_space_results.csv"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('single_space_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SingleSpaceProcessor:
    def __init__(self):
        self.client = None
        self.results = []
        self.processed_count = 0
        self.failed_count = 0
        self.processed_users = set()
        self.all_user_data = {}  # Store all posts for each user across files
        
        # Load previous progress if exists
        self.load_progress()
        
    def load_progress(self):
        """Load previous processing progress"""
        if os.path.exists(PROCESSED_USERS_FILE):
            try:
                with open(PROCESSED_USERS_FILE, 'r') as f:
                    data = json.load(f)
                    self.processed_users = set(data.get('processed_users', []))
                logger.info(f"Loaded {len(self.processed_users)} previously processed users")
            except Exception as e:
                logger.warning(f"Could not load previous progress: {e}")
        
        if os.path.exists(PROGRESS_FILE):
            try:
                with open(PROGRESS_FILE, 'r') as f:
                    progress = json.load(f)
                logger.info(f"Previous session: {progress}")
            except Exception as e:
                logger.warning(f"Could not load progress file: {e}")
    
    def save_progress(self, processed_files, current_file=None):
        """Save current processing progress"""
        progress = {
            'processed_files': processed_files,
            'current_file': current_file,
            'processed_users_count': len(self.processed_users),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress, f, indent=2)
        
        with open(PROCESSED_USERS_FILE, 'w') as f:
            json.dump({'processed_users': list(self.processed_users)}, f)
        
        logger.info(f"Progress saved: {len(self.processed_users)} users processed")
    
    def connect_to_space(self):
        """Connect to HuggingFace space"""
        logger.info(f"Connecting to HF Space: {SPACE}")
        try:
            self.client = Client(SPACE)
            logger.info("Successfully connected to HF Space")
        except Exception as e:
            logger.error(f"Failed to connect to Space: {e}")
            raise
    
    def load_csv_files(self, start_file=0, end_file=None):
        """Load specific range of CSV files"""
        csv_files = glob.glob(os.path.join(LARGE_DATASET_PATH, "*.csv"))
        csv_files.sort()
        
        if end_file is None:
            end_file = len(csv_files)
        
        files_to_process = csv_files[start_file:end_file]
        
        logger.info(f"Loading CSV files {start_file} to {end_file-1} ({len(files_to_process)} files)")
        
        all_posts = []
        
        for i, file_path in enumerate(files_to_process):
            file_name = os.path.basename(file_path)
            logger.info(f"Processing file {start_file + i}: {file_name}")
            
            try:
                # Read file in chunks
                chunk_count = 0
                file_rows = 0
                for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE):
                    chunk_count += 1
                    chunk_rows = len(chunk)
                    file_rows += chunk_rows
                    
                    # Filter required columns
                    required_cols = ['post_owner.id', 'surface.id', 'text', 'post_owner.name']
                    available_cols = [col for col in required_cols if col in chunk.columns]
                    
                    if 'post_owner.id' not in available_cols or 'text' not in available_cols:
                        continue
                    
                    chunk_filtered = chunk[available_cols].copy()
                    chunk_filtered = chunk_filtered.dropna(subset=['post_owner.id', 'text'])
                    chunk_filtered = chunk_filtered[chunk_filtered['text'].str.strip() != '']
                    
                    all_posts.append(chunk_filtered)
                    
                logger.info(f"  Loaded {file_rows:,} rows from {file_name}")
                
            except Exception as e:
                logger.error(f"Error processing {file_name}: {e}")
                continue
        
        if not all_posts:
            raise ValueError("No valid data found in specified CSV files")
        
        combined_df = pd.concat(all_posts, ignore_index=True)
        logger.info(f"Total posts loaded from files {start_file}-{end_file-1}: {len(combined_df):,}")
        
        return combined_df, files_to_process
    
    def load_sentiment_data(self):
        """Load sentiment analysis results"""
        logger.info("Loading sentiment analysis results...")
        sentiment_df = pd.read_csv(SENTIMENT_FILE)
        logger.info(f"Loaded {len(sentiment_df):,} sentiment records")
        return sentiment_df
    
    def accumulate_user_data(self, posts_df, sentiment_df):
        """Accumulate user data across multiple file processing sessions"""
        logger.info("Merging posts with sentiment data...")
        
        # Merge on surface.id
        merged_df = posts_df.merge(
            sentiment_df, 
            left_on='surface.id', 
            right_on='surface.id', 
            how='inner'
        )
        
        logger.info(f"Successfully merged: {len(merged_df):,} records")
        
        # Clean merged data
        merged_df = merged_df.dropna(subset=['post_owner.id', 'text', 'sentiment_overall'])
        merged_df = merged_df[merged_df['text'].str.strip() != '']
        
        logger.info(f"After cleaning: {len(merged_df):,} valid rows")
        
        # Accumulate user data
        new_users = 0
        updated_users = 0
        
        for user_id in merged_df['post_owner.id'].unique():
            user_posts = merged_df[merged_df['post_owner.id'] == user_id]
            
            if user_id not in self.all_user_data:
                self.all_user_data[user_id] = {
                    'posts': [],
                    'sentiments': [],
                    'surface_ids': [],
                    'user_name': user_posts['post_owner.name'].iloc[0] if 'post_owner.name' in user_posts.columns else 'Unknown'
                }
                new_users += 1
            else:
                updated_users += 1
            
            # Add new posts for this user
            for _, row in user_posts.iterrows():
                self.all_user_data[user_id]['posts'].append(row['text'])
                self.all_user_data[user_id]['sentiments'].append(row['sentiment_overall'])
                self.all_user_data[user_id]['surface_ids'].append(row['surface.id'])
        
        logger.info(f"User data updated: {new_users} new users, {updated_users} existing users updated")
        logger.info(f"Total users accumulated: {len(self.all_user_data)}")
        
        return len(self.all_user_data)
    
    def get_ready_users(self):
        """Get users that are ready for classification (excluding already processed)"""
        ready_users = []
        
        for user_id, user_data in self.all_user_data.items():
            if user_id not in self.processed_users:
                # Apply voting mechanism
                sentiments = user_data['sentiments']
                sentiment_counts = pd.Series(sentiments).value_counts()
                
                winning_sentiment = sentiment_counts.index[0]
                winning_count = sentiment_counts.iloc[0]
                total_count = len(sentiments)
                
                confidence = winning_count / total_count
                winning_percentage = confidence * 100
                is_unanimous = len(sentiment_counts) == 1
                
                user_record = {
                    'user_id': user_id,
                    'user_name': user_data['user_name'],
                    'num_posts': len(user_data['posts']),
                    'voted_sentiment': winning_sentiment,
                    'sentiment_confidence': confidence,
                    'winning_percentage': winning_percentage,
                    'is_unanimous': is_unanimous,
                    'surface_ids_sample': str(user_data['surface_ids'][:3]),
                    'individual_sentiments_sample': str(sentiments[:3]),
                    'sentiment_pos': (pd.Series(sentiments) == 'pos').sum(),
                    'sentiment_neg': (pd.Series(sentiments) == 'neg').sum(),
                    'first_post_sample': user_data['posts'][0][:200] + '...' if len(user_data['posts'][0]) > 200 else user_data['posts'][0]
                }
                
                ready_users.append(user_record)
        
        logger.info(f"Users ready for classification: {len(ready_users)} (excluding {len(self.processed_users)} already processed)")
        return ready_users
    
    def classify_user(self, user_data):
        """Classify a single user"""
        try:
            user_id = user_data['user_id']
            user_name = user_data['user_name']
            voted_sentiment = user_data['voted_sentiment']
            num_posts = user_data['num_posts']
            confidence = user_data['sentiment_confidence']
            first_post = user_data['first_post_sample']
            
            logger.info(f"Classifying user {user_id} ({user_name}): {num_posts} posts, sentiment: {voted_sentiment} ({confidence:.2f} confidence)")
            
            # Format post with sentiment context
            enhanced_post = f"[POST_SENTIMENT: {voted_sentiment}] [USER_SENTIMENT: {voted_sentiment}] {first_post}"
            
            # Prepare parameters for HF Space
            params = {f'param_{i}': '' for i in range(20)}
            params['param_0'] = enhanced_post
            
            # Call HF Space
            result = self.client.predict(**params, api_name='/predict')
            
            self.processed_count += 1
            self.processed_users.add(user_id)  # Mark as processed
            
            # Create result record
            result_record = user_data.copy()
            result_record['classification'] = result
            result_record['status'] = 'success'
            result_record['error'] = None
            result_record['processed_by'] = SPACE
            result_record['timestamp'] = datetime.now().isoformat()
            
            self.results.append(result_record)
            
            logger.info(f"Successfully classified user {user_id}")
            logger.info(f"Progress: {self.processed_count} users completed")
            
        except Exception as e:
            self.failed_count += 1
            
            # Create failure record
            result_record = user_data.copy()
            result_record['classification'] = None
            result_record['status'] = 'failed'
            result_record['error'] = str(e)
            result_record['processed_by'] = SPACE
            result_record['timestamp'] = datetime.now().isoformat()
            
            self.results.append(result_record)
            
            logger.error(f"Failed to classify user {user_data['user_id']}: {e}")
    
    def process_users(self, ready_users):
        """Process all ready users sequentially"""
        if not ready_users:
            logger.info("No users ready for processing")
            return
        
        logger.info(f"Processing {len(ready_users)} ready users sequentially")
        
        start_time = time.time()
        
        # Process users in batches for progress tracking and saving
        for batch_start in range(0, len(ready_users), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(ready_users))
            batch_users = ready_users[batch_start:batch_end]
            
            batch_num = (batch_start // BATCH_SIZE) + 1
            total_batches = (len(ready_users) + BATCH_SIZE - 1) // BATCH_SIZE
            
            logger.info(f"\n🔄 Processing batch {batch_num}/{total_batches} (users {batch_start + 1}-{batch_end})")
            
            batch_start_time = time.time()
            
            # Process each user in the batch
            for user_data in batch_users:
                self.classify_user(user_data)
                
                # Progress update every 10 users
                if self.processed_count % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / self.processed_count
                    remaining = len(ready_users) - self.processed_count
                    eta = remaining * avg_time
                    logger.info(f"Progress: {self.processed_count}/{len(ready_users)} users, ETA: {eta/60:.1f} minutes")
            
            batch_time = time.time() - batch_start_time
            logger.info(f"✅ Batch {batch_num} completed in {batch_time:.1f} seconds")
            
            # Save progress after each batch
            self.save_progress_after_batch()
        
        total_time = time.time() - start_time
        logger.info(f"\n🎉 Completed processing {len(ready_users)} users in {total_time/60:.1f} minutes")
    
    def save_progress_after_batch(self):
        """Save progress after each batch"""
        if self.results:
            # Append to results file
            results_df = pd.DataFrame(self.results)
            
            if os.path.exists(RESULTS_FILE):
                # Append to existing file
                results_df.to_csv(RESULTS_FILE, mode='a', header=False, index=False)
            else:
                # Create new file
                results_df.to_csv(RESULTS_FILE, index=False)
            
            logger.info(f"Saved {len(self.results)} new results to {RESULTS_FILE}")
            self.results = []  # Clear results to save memory
    
    def run_incremental(self, start_file=0, end_file=None):
        """Main incremental processing pipeline"""
        logger.info("Starting single-space incremental processing pipeline")
        logger.info("="*60)
        
        start_time = time.time()
        
        try:
            # Connect to space
            self.connect_to_space()
            
            # Load sentiment data once
            sentiment_df = self.load_sentiment_data()
            
            # Load and accumulate data from specified CSV files
            posts_df, processed_files = self.load_csv_files(start_file, end_file)
            total_users = self.accumulate_user_data(posts_df, sentiment_df)
            
            # Get users ready for classification
            ready_users = self.get_ready_users()
            
            if ready_users:
                # Process ready users
                self.process_users(ready_users)
                
                # Save final progress
                self.save_progress([f.split('/')[-1] for f in processed_files])
            
            # Summary
            total_time = time.time() - start_time
            
            logger.info("\n" + "="*70)
            logger.info("SINGLE-SPACE INCREMENTAL PROCESSING SUMMARY")
            logger.info("="*70)
            logger.info(f"Files processed: {len(processed_files)}")
            logger.info(f"File range: {start_file} to {end_file-1 if end_file else 'end'}")
            logger.info(f"Total users in system: {total_users}")
            logger.info(f"Users processed this session: {self.processed_count}")
            logger.info(f"Total processed users: {len(self.processed_users)}")
            logger.info(f"Failed classifications: {self.failed_count}")
            logger.info(f"Success rate: {self.processed_count/(self.processed_count + self.failed_count)*100:.1f}%" if (self.processed_count + self.failed_count) > 0 else "N/A")
            logger.info(f"Processing time: {total_time/60:.1f} minutes")
            logger.info(f"Average time per user: {total_time/max(self.processed_count, 1):.2f} seconds")
            logger.info(f"Results file: {RESULTS_FILE}")
            
            return RESULTS_FILE
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

def main():
    global BATCH_SIZE
    
    parser = argparse.ArgumentParser(description='Process CSV files incrementally with single space')
    parser.add_argument('--start', type=int, default=0, help='Start file index (0-based)')
    parser.add_argument('--end', type=int, help='End file index (exclusive)')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help=f'Batch size (default: {BATCH_SIZE})')
    
    args = parser.parse_args()
    
    if args.batch_size:
        BATCH_SIZE = args.batch_size
    
    processor = SingleSpaceProcessor()
    processor.run_incremental(start_file=args.start, end_file=args.end)

if __name__ == "__main__":
    main()