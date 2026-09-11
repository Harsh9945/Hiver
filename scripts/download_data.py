# scripts/download_data.py
import os, sys

KAGGLE_DATASET = 'thoughtvector/customer-support-on-twitter'
RAW_DIR = 'data/raw'
RAW_FILE = os.path.join(RAW_DIR, 'twcs.csv')

def main():
    print('=' * 75)
    print('   HIVER SUPPORT AGENT: RAW TWCS DATASET DOWNLOAD HELPER')
    print('=' * 75)
    if os.path.exists(RAW_FILE):
        sz_mb = os.path.getsize(RAW_FILE) / (1024 * 1024)
        print(f'\nRaw dataset already exists at {RAW_FILE} ({sz_mb:.2f} MB).')
        print('No download needed. Run python src/ingest.py to re-process.')
        return

    print('\nThe raw TWCS dataset is ~500 MB (2.81 million customer support tweets).')
    print('NOTE: You do NOT need this file to run run_tests.py or eval/run_eval.py.')
    print('The repository includes the verified held-out benchmark and retrieval index.\n')
    print('Option 1: Direct Kaggle CLI download:')
    print(f'  kaggle datasets download -d {KAGGLE_DATASET} -p {RAW_DIR} --unzip\n')
    print('Option 2: Web Browser:')
    print('  1. Visit: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter')
    print(f'  2. Download twcs.csv and place it in {RAW_DIR}/twcs.csv\n')

if __name__ == '__main__':
    main()
