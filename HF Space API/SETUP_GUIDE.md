# Configuration Guide for Team Collaboration

### 1. Update Paths in `single_space_processor.py`

Open `single_space_processor.py` and update these lines (around line 17-21):

```python
# Configuration - UPDATE THESE PATHS FOR YOUR SYSTEM
LARGE_DATASET_PATH = "C:/path/to/your/ProfilingSocialMedia/"  # Path to your 21 CSV files
SENTIMENT_FILE = "user_sentiment_from_blocks.csv"  # Keep this as is
SPACE = "CatoEr/Profiling-Social-Media-Users"  # Use CatoEr space for team member 2
```

### 2. File Structure Setup
```
Your Project Folder/
├── profiling-users/
│   └── HF Space API/
│       ├── single_space_processor.py
│       ├── user_sentiment_from_blocks.csv
│       └── requirements.txt
└── ProfilingSocialMedia/
    ├── islam_2024_01.csv
    ├── islam_2024_02.csv
    └── ... (all 21 CSV files)
```

### 3. Environment Setup (Windows)
```cmd
# Create virtual environment
python -m venv profiling_env
profiling_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Test Setup
```cmd
# Test with first 2 files
python single_space_processor.py --start 0 --end 2
```

### 5. Your Processing Assignment
```cmd
# Process files 11-20 (second half of dataset)
python single_space_processor.py --start 11 --end 21
```

