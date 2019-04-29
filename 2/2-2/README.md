## HW 2-2 Chatbot

[Download Data Set](https://drive.google.com/file/d/1C0AGRNOuEX9OwYaYkUFM8tZr1dBwbOhM/view?usp=sharing)

This is a chatbot model using LSTM model and Attention. The model is the same as HW2-1



To generate dictionary.json(a json file containing the dictionary of word and one-hot encoding number)

```bash
python dictionary.py
```



To generate txt files that filtered unwanted sentence

```
python Generate_txt.py
```



To load pre-trained model or start training  a model

```
python att(_2layer)_load.py
```

In fact, 2 layer model has worse performance than the original one



To test  a model

```
python att(_2layer)_test.py
```

In fact, 2 layer model has worse performance than the original one





The work is implement on python 3.7.2, keras(2.2.4), h5py(2.9.0), numpy(1.16.2), matplotlib(3.0.3).

