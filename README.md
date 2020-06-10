# Taris
Transformer-based online speech recognition system with TensorFlow 2.0

### About

Taris is an approach to online speech recognition described in [1].
The system dynamically segments a spoken sentence by learning to count the number of spoken words therein.
Decoding is conditioned on a dynamic window of segments, instead of an entire utterance as in the original sequence to sequence architecture.

This repository also maintains the audio-visual alignment and fusion strategy AV Align [2,3] currently implemented with the Transformer stacks instead of the original recurrent networks [4]


### How to use

### References

[1] Learning to Count Words in Fluent Speech enables Online Speech Recognition\
George Sterpu, Christian Saam, Naomi Harte\
Under review\
[pdf](https://github.com/georgesterpu/georgesterpu.github.io/raw/master/papers/gg2020.pdf)

[2] How to Teach DNNs to Pay Attention to the Visual Modality in Speech Recognition\
George Sterpu, Christian Saam, Naomi Harte\
IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2020\
[pdf accepted version](https://raw.githubusercontent.com/georgesterpu/georgesterpu.github.io/master/papers/taslp2020.pdf) [IEEE version](https://ieeexplore.ieee.org/document/9035650)

[3] Attention-based Audio-Visual Fusion for Robust Automatic Speech Recognition\
George Sterpu, Christian Saam, Naomi Harte\
in ICMI 2018
[pdf](https://arxiv.org/pdf/1809.01728.pdf)

[4] Should we hard-code the recurrence concept or learn it instead ?
Exploring the Transformer architecture for Audio-Visual Speech Recognition \
George Sterpu, Christian Saam, Naomi Harte\
Under review\
[pdf](https://arxiv.org/pdf/2005.09297.pdf)
### Dependencies
```
tensorflow
```

