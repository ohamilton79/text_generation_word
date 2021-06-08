# Word level text generation

Using Keras to generate text word by word.

## Dataset
The dataset used is Oliver Twist by Charles Dickens, stored in the `corpus.txt` file and available at the following link:
https://www.gutenberg.org/files/730/730-0.txt

## Dependencies
The only dependencies are Tensorflow, Keras and h5py. Run the following:
```bash
pip install tensorflow keras 'h5py<3.0.0'
```
## Training
* When training the model, the word-to-integer mapping, tokenizer, and weights files will be stored in the `data` directory. 
* **WARNING**: This process will overwrite existing pre-trained weights.
* Run the following command to train the model:
```bash
python rnn_train.py
```

## Testing
To test the model on the testing data provided by the IMDb dataset, run the following, specifying:
* The location of the weights file you want to use relative to the working directory:
(`data/weights-500.hdf5` is the recommended value for the `path_to_weights_file` parameter)
```bash
python manual_test.py 'path_to_weights_file'
```

# Examples
```python
>>> python manual_test.py data/weights-500.hdf5
Generated text with temperature 0.2:
warmly . “ if he is not , ” said mr . grimwig , “ i ’ ll — ” and down went the stick . “ i ’ ll say it , ” replied the matron , “ how ’ a long of the book of it in the , , and you to be our of life ; and that ’ s be that , my dear , what the board got away in , ” a man is opened in a for three morning , that it must be these things that was being be . ”
Generated text with temperature 0.25:
warmly . “ if he is not , ” said mr . grimwig , “ i ’ ll — ” and down went the stick . “ i ’ ll bear it , ” replied the matron , “ how ’ s a vagrant . ” “ it is you , sir , ” cried the girl . “ yes , it , ” replied noah . the jew was opened ; on the room of an exclamations of “ never made you , he spoke , ma ’ oliver , he is by the thing and was away ,
Generated text with temperature 0.3:
warmly . “ if he is not , ” said mr . grimwig , “ i ’ ll — ” and down went the stick . “ i ’ ll bear it , ” replied the matron , “ how ’ s a be ; to know that it ’ s better a deal and two is ? or it ’ s all it of be it in a be , to the same to a secret two . it must have the book up again in the town . ” “ ” “ the girl ’ s a man
Generated text with temperature 0.35:
warmly . “ if he is not , ” said mr . grimwig , “ i ’ ll — ” and down went the stick . “ i ’ ll am very about , sir , ” replied mr . bumble . “ help , ” replied oliver , “ what ’ ll you ? ” “ mr . bumble , no , by the one of my kind of it . “ no , sir , ” replied the matron , “ to him . ” “ oh ! ” cried the gentleman ; “ to out of the
Generated text with temperature 0.4:
warmly . “ if he is not , ” said mr . grimwig , “ i ’ ll — ” and down went the stick . “ i ’ ll bear it , ” replied the matron , “ how bill it is , ” replied nancy , the mr . bumble ; “ for the one of some pounds that he had taken to , to think it ; the boy ’ s set the gold of one of it ; and his in the room was well were the fire of his face . “ ah ! ”
```
