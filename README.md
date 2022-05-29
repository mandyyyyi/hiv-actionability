# Baseline
We provide pretrained bert as baseline. 
## Train Model
```bash
python baseline/run_bert.py --trainData [dataset/train_file.csv] --tweetContent_train [content] --tweetLabel [label] --tweetID [tweet_ids] --tweetContent [content] --outputFile [dataset/bert_out.csv] --testData [dataset/inputFile.csv]

```
--trainData The directory for retrieving the train data
--tweetContent_train The column name for the tweet content in train file
--tweetLabel The column name for the tweet label in train file
--tweetID The column name for the tweet id in test file
--tweetContent The column name for the tweet content in test file
--outputFile The output prediction file of model
--testData The file to be predicted by the model (test file)