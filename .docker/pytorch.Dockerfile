FROM python
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN pip install pandas matplotlib scikit-learn networkx==2.8.8 node2vec==0.4.6
