python3 -m classifier.classifier --model RNN --type binary --granular statement
python3 -m classifier.classifier --model CNN --type binary --granular statement
python3 -m classifier.classifier --model LSTM --type binary --granular statement
python3 -m classifier.classifier --model multiDNN --type binary --granular statement
python3 -m classifier.classifier --model RF --type binary --granular statement

python3 -m classifier.classifier --model RNN --type binary --granular function
python3 -m classifier.classifier --model CNN --type binary --granular function
python3 -m classifier.classifier --model LSTM --type binary --granular function

python3 -m classifier.classifier --model RNN --type multiclass --granular statement
python3 -m classifier.classifier --model CNN --type multiclass --granular statement
python3 -m classifier.classifier --model LSTM --type multiclass --granular statement
python3 -m classifier.classifier --model multiDNN --type multiclass --granular statement
python3 -m classifier.classifier --model RF --type multiclass --granular statement

python3 -m classifier.classifier --model RNN --type multiclass --granular function
python3 -m classifier.classifier --model CNN --type multiclass --granular function
python3 -m classifier.classifier --model LSTM --type multiclass --granular function