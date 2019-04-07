# HW2-1 Video Caption

This is a MLDS practice implemented on python 3.6.8, keras(2.2.4), h5py(2.9.0), numpy(1.14.5), tensorflow-gpu(1.9.0).


[Download Data sets](<https://drive.google.com/drive/folders/1kwKemVE8CZqjv1e2p72JnL2eeWjKJOQ0?fbclid=IwAR3FeMaGQ7-B9KjE4cQvlN3FuElRnxOznpLVseERNwLOHNWeze9TpNlGf5E>)

To parse the datasets into json ditionaries:

```bash
python3 read.py
```



### HW2-1-1

Basic LSTM model
![img](https://lh4.googleusercontent.com/pjdRBSq0my7Em9qE1sm8ug624jOYbuqh5_bfe13n6Xe5-a6dFQqjHLZYy-M9QIcJs2KRD2wkM3z3JdpRhFJx7LPpZGuA3BA9CSTY0CO3vIoUjqvjPf2q83N9Ae9w5b9yVnmX8LsxFa4)

The dimension of input of input_2 and output of  dense_1 should be (None, None, 4096) (None, None, 2497).

To train and test the model on training and testing sets:

```
python3 s2s.py
```

P.S. You can manually change the EPOCH, and latent_dim by yourself to varify the model.





### HW2-1-2

Model description:

![img](https://lh4.googleusercontent.com/U6-KabWwIJtCNgy-c37ueG99i9PrbGqsyXdjLdxxtZi25Z60to-2f0NOsgJUhvgNZjAO2qIYnbQOrqjqSzH1bt5-GpEddmOcXDdXxXb7jh6KZVye0h0Lg3dHtMh--RTjJl65Rzrj0xA)

The model is based on [*Effective Approaches to Attention-based Neural Machine Translation*](<https://arxiv.org/pdf/1508.04025.pdf>) , by Minh-Thang Luong Hieu Pham Christopher D. Manning Computer Science Department, Stanford University, Stanford



To train and test the model on training and testing sets:

```
python3 s2s-att.py
```
