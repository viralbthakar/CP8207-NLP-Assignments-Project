# Assignment 2

## Create Custom Word2Vec
```
python custom_word2vec.py \
    --csv-file-paths ../data/raw_csvs/book-raw-paragraphs.csv \
    ../data/raw_csvs/characters-raw-paragraphs.csv \
    ../data/raw_csvs/dragons-raw-paragraphs.csv \
    ../data/raw_csvs/episodes-raw-paragraphs.csv \
    --logdir ../data/word2vec/logs/exp-01-all-data-w4-nng4-ed128 \
    --window-size 3 \
    --num-negs 4 \
    --batch-size 1024 \
    --epochs 20 \
    --embedding-dim 128
```

## Topic Comparison and Interpretation

