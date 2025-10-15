#!/usr/bin/env python3
"""
Voting mechanism exploration script
Analyzes how the voting system will work with your data
Shows the relationship between post_owner.id, surface.id, and sentiment votes
"""

import pandas as pd
import numpy as np
from collections import Counter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def explore_voting_mechanism(posts_csv_path: str, sentiment_csv_path: str):
    """Explore how the voting mechanism will work with your data"""
    
    print("="*70)
    print("VOTING MECHANISM EXPLORATION")
    print("="*70)
    
    try:
        # Load datasets
        logger.info("Loading datasets...")
        posts_df = pd.read_csv(posts_csv_path)
        sentiment_df = pd.read_csv(sentiment_csv_path)
        
        print(f"📊 Posts dataset: {len(posts_df):,} records")
        print(f"📊 Sentiment dataset: {len(sentiment_df):,} records")
        
        # Merge data
        merged_df = pd.merge(posts_df, sentiment_df, on='surface.id', how='inner')
        print(f"📊 Merged dataset: {len(merged_df):,} records")
        
        # Clean data
        clean_df = merged_df.dropna(subset=['post_owner.id', 'post_owner.name', 'text'])
        clean_df = clean_df[clean_df['text'].str.strip() != '']
        print(f"📊 Clean dataset: {len(clean_df):,} records")
        
        # Analyze user-post relationship
        print(f"\n👥 USER-POST ANALYSIS:")
        user_post_counts = clean_df['post_owner.id'].value_counts()
        print(f"   - Unique users (post_owner.id): {len(user_post_counts):,}")
        print(f"   - Posts per user (average): {user_post_counts.mean():.1f}")
        print(f"   - Posts per user (median): {user_post_counts.median():.1f}")
        print(f"   - Max posts by single user: {user_post_counts.max():,}")
        
        # Analyze users with multiple posts (voting candidates)
        multi_post_users = user_post_counts[user_post_counts > 1]
        single_post_users = user_post_counts[user_post_counts == 1]
        
        print(f"\n🗳️  VOTING ANALYSIS:")
        print(f"   - Users with multiple posts (need voting): {len(multi_post_users):,} ({len(multi_post_users)/len(user_post_counts)*100:.1f}%)")
        print(f"   - Users with single post (no voting needed): {len(single_post_users):,} ({len(single_post_users)/len(user_post_counts)*100:.1f}%)")
        
        # Analyze sentiment consistency for multi-post users
        sentiment_consistency_analysis = []
        
        for user_id in multi_post_users.index[:100]:  # Analyze first 100 multi-post users
            user_posts = clean_df[clean_df['post_owner.id'] == user_id]
            sentiments = user_posts['sentiment_overall'].tolist()
            sentiment_counts = Counter(sentiments)
            
            total_posts = len(sentiments)
            unique_sentiments = len(sentiment_counts)
            most_common_sentiment = sentiment_counts.most_common(1)[0]
            winning_sentiment = most_common_sentiment[0]
            winning_count = most_common_sentiment[1]
            confidence = winning_count / total_posts
            
            sentiment_consistency_analysis.append({
                'user_id': user_id,
                'total_posts': total_posts,
                'unique_sentiments': unique_sentiments,
                'winning_sentiment': winning_sentiment,
                'winning_count': winning_count,
                'confidence': confidence,
                'is_unanimous': unique_sentiments == 1,
                'sentiment_distribution': dict(sentiment_counts)
            })
        
        # Voting quality statistics
        unanimous_users = sum(1 for analysis in sentiment_consistency_analysis if analysis['is_unanimous'])
        high_confidence_users = sum(1 for analysis in sentiment_consistency_analysis if analysis['confidence'] >= 0.7)
        medium_confidence_users = sum(1 for analysis in sentiment_consistency_analysis if 0.5 <= analysis['confidence'] < 0.7)
        low_confidence_users = sum(1 for analysis in sentiment_consistency_analysis if analysis['confidence'] < 0.5)
        
        total_analyzed = len(sentiment_consistency_analysis)
        
        print(f"\n📈 VOTING QUALITY (sample of {total_analyzed} multi-post users):")
        print(f"   - Unanimous sentiment: {unanimous_users} ({unanimous_users/total_analyzed*100:.1f}%)")
        print(f"   - High confidence (≥70%): {high_confidence_users} ({high_confidence_users/total_analyzed*100:.1f}%)")
        print(f"   - Medium confidence (50-69%): {medium_confidence_users} ({medium_confidence_users/total_analyzed*100:.1f}%)")
        print(f"   - Low confidence (<50%): {low_confidence_users} ({low_confidence_users/total_analyzed*100:.1f}%)")
        
        # Show examples of different voting scenarios
        print(f"\n🔍 VOTING EXAMPLES:")
        
        # Find examples of different scenarios
        unanimous_example = next((a for a in sentiment_consistency_analysis if a['is_unanimous'] and a['total_posts'] > 2), None)
        high_conf_example = next((a for a in sentiment_consistency_analysis if a['confidence'] >= 0.8 and not a['is_unanimous']), None)
        low_conf_example = next((a for a in sentiment_consistency_analysis if a['confidence'] < 0.6), None)
        
        if unanimous_example:
            user_id = unanimous_example['user_id']
            user_posts = clean_df[clean_df['post_owner.id'] == user_id]
            print(f"\n   1️⃣ UNANIMOUS EXAMPLE (User {user_id}):")
            print(f"      Posts: {unanimous_example['total_posts']}")
            print(f"      Sentiment: {unanimous_example['winning_sentiment']} (100% confidence)")
            print(f"      Sample posts:")
            for i, (_, post) in enumerate(user_posts.head(3).iterrows()):
                print(f"         - [{post['sentiment_overall']}] {post['text'][:60]}...")
        
        if high_conf_example:
            user_id = high_conf_example['user_id']
            user_posts = clean_df[clean_df['post_owner.id'] == user_id]
            print(f"\n   2️⃣ HIGH CONFIDENCE EXAMPLE (User {user_id}):")
            print(f"      Posts: {high_conf_example['total_posts']}")
            print(f"      Winning sentiment: {high_conf_example['winning_sentiment']} ({high_conf_example['confidence']:.0%} confidence)")
            print(f"      Distribution: {high_conf_example['sentiment_distribution']}")
            print(f"      Sample posts:")
            for i, (_, post) in enumerate(user_posts.head(3).iterrows()):
                print(f"         - [{post['sentiment_overall']}] {post['text'][:60]}...")
        
        if low_conf_example:
            user_id = low_conf_example['user_id']
            user_posts = clean_df[clean_df['post_owner.id'] == user_id]
            print(f"\n   3️⃣ LOW CONFIDENCE EXAMPLE (User {user_id}):")
            print(f"      Posts: {low_conf_example['total_posts']}")
            print(f"      Winning sentiment: {low_conf_example['winning_sentiment']} ({low_conf_example['confidence']:.0%} confidence)")
            print(f"      Distribution: {low_conf_example['sentiment_distribution']}")
            print(f"      Sample posts:")
            for i, (_, post) in enumerate(user_posts.head(3).iterrows()):
                print(f"         - [{post['sentiment_overall']}] {post['text'][:60]}...")
        
        # Overall sentiment distribution after voting
        print(f"\n📊 PREDICTED SENTIMENT DISTRIBUTION AFTER VOTING:")
        winning_sentiments = [analysis['winning_sentiment'] for analysis in sentiment_consistency_analysis]
        overall_distribution = Counter(winning_sentiments)
        
        for sentiment, count in overall_distribution.most_common():
            percentage = count / len(winning_sentiments) * 100
            print(f"   - {sentiment}: {count} users ({percentage:.1f}%)")
        
        # Surface.id to post_owner.id mapping analysis
        print(f"\n🔗 SURFACE.ID MAPPING ANALYSIS:")
        surface_to_owner = clean_df.groupby('surface.id')['post_owner.id'].nunique()
        owner_to_surface = clean_df.groupby('post_owner.id')['surface.id'].nunique()
        
        print(f"   - Surface IDs with multiple owners: {(surface_to_owner > 1).sum()}")
        print(f"   - Average surfaces per user: {owner_to_surface.mean():.1f}")
        print(f"   - Users with multiple surface IDs: {(owner_to_surface > 1).sum()}")
        
        # API format preview
        print(f"\n🚀 API FORMAT PREVIEW:")
        if high_conf_example:
            user_id = high_conf_example['user_id']
            user_posts = clean_df[clean_df['post_owner.id'] == user_id]
            winning_sentiment = high_conf_example['winning_sentiment']
            confidence = high_conf_example['confidence']
            
            print(f"   Example user {user_id} will be sent to HF Space as:")
            print(f"   Voted sentiment: {winning_sentiment} (confidence: {confidence:.0%})")
            
            for i, (_, post) in enumerate(user_posts.head(3).iterrows()):
                individual_sentiment = post['sentiment_overall']
                post_text = post['text'][:50]
                combined = f"[POST_SENTIMENT: {individual_sentiment}] [USER_SENTIMENT: {winning_sentiment}] {post_text}..."
                print(f"      param_{i}: {combined}")
        
        # Save detailed analysis
        analysis_df = pd.DataFrame(sentiment_consistency_analysis)
        analysis_output = "voting_mechanism_analysis.csv"
        analysis_df.to_csv(analysis_output, index=False)
        print(f"\n💾 Detailed voting analysis saved to: {analysis_output}")
        
        return clean_df, sentiment_consistency_analysis
        
    except Exception as e:
        logger.error(f"Error during voting exploration: {str(e)}")
        print(f"❌ Error: {str(e)}")
        return None, None

def generate_voting_recommendations(sentiment_consistency_analysis):
    """Generate recommendations based on voting analysis"""
    
    print(f"\n" + "="*70)
    print("VOTING MECHANISM RECOMMENDATIONS")
    print("="*70)
    
    if not sentiment_consistency_analysis:
        print("❌ No analysis data available for recommendations")
        return
    
    total_users = len(sentiment_consistency_analysis)
    unanimous_count = sum(1 for a in sentiment_consistency_analysis if a['is_unanimous'])
    high_conf_count = sum(1 for a in sentiment_consistency_analysis if a['confidence'] >= 0.7)
    low_conf_count = sum(1 for a in sentiment_consistency_analysis if a['confidence'] < 0.5)
    
    unanimous_pct = unanimous_count / total_users * 100
    high_conf_pct = high_conf_count / total_users * 100
    low_conf_pct = low_conf_count / total_users * 100
    
    print(f"📋 QUALITY ASSESSMENT:")
    
    if unanimous_pct > 60:
        print(f"   ✅ EXCELLENT: {unanimous_pct:.1f}% of users have unanimous sentiment")
        print(f"      → High quality voting results expected")
    elif unanimous_pct > 40:
        print(f"   ✅ GOOD: {unanimous_pct:.1f}% of users have unanimous sentiment")
        print(f"      → Voting will improve classification quality")
    else:
        print(f"   ⚠️  MODERATE: Only {unanimous_pct:.1f}% of users have unanimous sentiment")
        print(f"      → Voting is necessary but results may be mixed")
    
    if high_conf_pct > 70:
        print(f"   ✅ HIGH CONFIDENCE: {high_conf_pct:.1f}% of users have ≥70% sentiment agreement")
    else:
        print(f"   ⚠️  MIXED CONFIDENCE: Only {high_conf_pct:.1f}% of users have ≥70% sentiment agreement")
    
    if low_conf_pct > 20:
        print(f"   ⚠️  ATTENTION NEEDED: {low_conf_pct:.1f}% of users have <50% confidence")
        print(f"      → Consider filtering or special handling for these users")
    
    print(f"\n💡 RECOMMENDATIONS:")
    
    # Confidence threshold recommendations
    print(f"   1️⃣ CONFIDENCE THRESHOLDS:")
    print(f"      → Include all users: Get maximum coverage")
    print(f"      → Include only ≥70% confidence: Higher quality, {100-high_conf_pct:.1f}% data loss")
    print(f"      → Include only unanimous: Highest quality, {100-unanimous_pct:.1f}% data loss")
    
    # API format recommendations
    print(f"   2️⃣ API FORMAT STRATEGY:")
    print(f"      → OPTION A: Use voted sentiment for all posts")
    print(f"        Format: [USER_SENTIMENT: {'{voted_sentiment}'} ({'{confidence}'})] {'{post_text}'}")
    print(f"      → OPTION B: Use both individual and voted sentiment (RECOMMENDED)")
    print(f"        Format: [POST_SENTIMENT: {'{individual}'}] [USER_SENTIMENT: {'{voted}'}] {'{post_text}'}")
    
    # Processing recommendations
    print(f"   3️⃣ PROCESSING STRATEGY:")
    if total_users < 1000:
        print(f"      → Use standard processing (voting_sentiment_classifier.py)")
    else:
        print(f"      → Use batch processing for {total_users} users")
        print(f"      → Consider processing high-confidence users first")
    
    print(f"   4️⃣ QUALITY CONTROL:")
    print(f"      → Monitor classification success rate by confidence level")
    print(f"      → Consider separate analysis for unanimous vs. voted users")
    print(f"      → Flag low-confidence classifications for manual review")

def main():
    """Main function for voting exploration"""
    
    # File paths
    POSTS_CSV = "Facebook Social Media Sample 2.csv"
    SENTIMENT_CSV = "user_sentiment_from_blocks.csv"
    
    # Check files exist
    import os
    if not os.path.exists(POSTS_CSV):
        print(f"❌ Posts file not found: {POSTS_CSV}")
        return
        
    if not os.path.exists(SENTIMENT_CSV):
        print(f"❌ Sentiment file not found: {SENTIMENT_CSV}")
        return
    
    # Run exploration
    clean_df, analysis = explore_voting_mechanism(POSTS_CSV, SENTIMENT_CSV)
    
    if analysis:
        # Generate recommendations
        generate_voting_recommendations(analysis)
        
        print(f"\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("1. Review the voting mechanism analysis above")
        print("2. Check the voting_mechanism_analysis.csv file for detailed data")
        print("3. If the quality looks good, run the voting classifier:")
        print("   python voting_sentiment_classifier.py")
        print("4. Monitor confidence scores in the results")
        print("5. Consider filtering results by confidence level for final analysis")

if __name__ == "__main__":
    main()