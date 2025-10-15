# Social Media User Profiling with Sentiment Analysis

A comprehensive system for classifying social media users across multiple demographic categories (Race, Age, Gender, Education, Sexual Orientation) using sentiment analysis and voting mechanisms.

## 🎯 Project Overview

This project processes large-scale social media datasets to:
1. **Combine sentiment analysis** with social media posts using democratic voting
2. **Handle multi-post users** through intelligent voting mechanisms  
3. **Classify users demographically** using HuggingFace ML models
4. **Support collaborative processing** across multiple team members

## 📊 Dataset Structure

### Required Files
- **21 CSV files** (~2GB total): `islam_2024_01.csv` to `islam_2024_12_02.csv`
- **Sentiment data**: `user_sentiment_from_blocks.csv` (182K+ sentiment records)

### Expected CSV Columns
```
post_owner.id, surface.id, text, post_owner.name, [other columns...]
```

### Key Statistics
- **1.2M+ posts** across 21 CSV files
- **194K+ unique users** to be classified
- **97.8% high confidence** voting results
- **95.8% unanimous sentiment** across multi-post users

## 🛠️ Setup Instructions

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv profiling_env
source profiling_env/bin/activate  # On Windows: profiling_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation
```
📁 Your Project Folder/
├── profiling-users/
│   └── HF Space API/                    # This repository
│       ├── single_space_processor.py   # Main processing script
│       ├── user_sentiment_from_blocks.csv
│       └── requirements.txt
└── ProfilingSocialMedia/               # Your CSV data folder
    ├── islam_2024_01.csv
    ├── islam_2024_02.csv
    └── ... (21 CSV files total)
```

### 3. HuggingFace Space Configuration
- **Team Member 1**: Uses `baranorhan/Profiling-Social-Media-Users`
- **Team Member 2**: Uses `CatoEr/Profiling-Social-Media-Users`

## 🚀 Usage Guide

### Single Space Processing (Recommended)

#### Quick Test (First 2 files)
```bash
python single_space_processor.py --start 0 --end 2
```

#### Process File Range
```bash
# Process files 0-10 (for team member 1)
python single_space_processor.py --start 0 --end 11

# Process files 11-20 (for team member 2)  
python single_space_processor.py --start 11 --end 21
```

#### Process All Files
```bash
python single_space_processor.py --start 0 --end 21
```

### Dual Space Processing (Advanced)
```bash
# Uses both spaces in parallel for faster processing
python large_dataset_processor.py --start 0 --end 21
```
