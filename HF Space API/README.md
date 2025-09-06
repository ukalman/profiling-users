# HF Space API - Social Media User Profiling

This folder contains scripts to test and analyze social media user profiling using Hugging Face Spaces API.

## Files

- `test_hf_space.py` - Main script to classify users using HF Space API
- `generate_summary.py` - Script to generate summary statistics from classification results
- `requirements.txt` - Python dependencies
- `Facebook Social Media Sample 2.csv` - Sample data file

## Setup

1. **Create and activate virtual environment:**
```bash
cd "HF Space API"
python3 -m venv hf_space_env
source hf_space_env/bin/activate  # On Windows: hf_space_env\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### 1. Classify Users

Run the main classification script:
```bash
python test_hf_space.py
```

This will:
- Load data from `../Islam 2024 Data.csv` (in parent directory)
- Group posts by user (`post_owner.name`)
- Send up to 20 texts per user to the HF Space API
- Save results to `../classification_results_islam_2024_01.csv`
- Generate log file `../hf_space_test.log`

### 2. Generate Summary Statistics

After classification, generate summary statistics:
```bash
python generate_summary.py
```

This will:
- Parse classification results from `../classification_results.csv`
- Generate summary CSVs in the parent directory:
  - Overall summary with all categories
  - Individual category summaries (Race, Age, Education, Gender, Sexual Orientation)
  - Detailed user breakdown

## Configuration

### test_hf_space.py
- `CSV_FILE_PATH`: Path to input CSV file
- `OUTPUT_FILE_PATH`: Path for classification results
- `DELAY_BETWEEN_REQUESTS`: Seconds to wait between API calls (rate limiting)

### generate_summary.py
- `RESULTS_CSV_PATH`: Path to classification results CSV
- `OUTPUT_DIR`: Directory to save summary files

## Output Files

### Classification Results
- Contains user-by-user classification results with probabilities and predictions

### Summary Files
- `classification_summary_overall_[timestamp].csv` - Combined summary of all categories
- `classification_summary_[category]_[timestamp].csv` - Individual category summaries
- `classification_detailed_breakdown_[timestamp].csv` - User-by-user predictions

## Example Summary Output

```
Category,Class,Count,Percentage,Average_Probability
RACE,Asian,10,71.43,65.55
RACE,White,2,14.29,18.87
SEXUAL ORIENTATION,Heterosexual,12,85.71,87.18
SEXUAL ORIENTATION,LGBTQ,2,14.29,12.82
```

## Notes

- All data files and results are stored in the parent directory
- The scripts use relative paths (`../`) to access files in the parent directory
- Rate limiting is implemented to avoid overwhelming the HF Space API
- Comprehensive logging is available for debugging
