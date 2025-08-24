---
layout: post
title: "Topic Modeling"
author: vuvietduy
categories: [NLP, Tutorial]
featured: false
published: true
image:
toc: true
excerpt: "Topic Modeling"
---

Section Goals

- Understand Topic Modeling
- Learn Latent Dirichlet Allocation
- Implement LDA
- Understand Non-Negative Matrix Factorization
- Implement NMF
- Apply LDA and NMF with a project

## 1. Topic Modeling

- Topic Modeling allows for us to efficiently analyze large volumes of text by clustering documents into topics
- A large amount of text data is unlabeled meaning we won't be able to apply our previous supervised learning approaches to create machine learning models for the data
- If we have unlabeled data, then we can attempt to discover labels.
- In the case of text data, this means attempting to discover clusters of documents, grouped together by topic
- A very important idea to keep in mind here is that we don't know the correct topic or right answer
- All we know is that the documents clustered together share similar topic ideas
- It is up to the user to indentify what these topics represent
- We will begin by examining how Latent Dirichlet Allocation can attempt to discover topics for a corpus of documents

## 2. Các loại Topic Modeling

Trước đây phương pháp $tf-idf$ được sử dụng khá phổ biến để mã hóa văn bản thành vector. Ta khởi tạo một tập hợp các từ (words hoặc terms) hay còn gọi là túi các từ (bag of word), tập hợp những từ này đã loại bỏ stop word. $tf-idf$ sẽ được tính bằng cách đo lường tần suất xuất hiện của từ trong văn bản chia cho tần suât văn bản mà có xuất hiện trên toàn bộ văn bản (corpus)

$w_i,_j = tf_i,_j x log\frac{N}{df_j}$

- $w_i,_j$: $tf-idf$ score
- $tf_i,_j$: occurrences of term in document
- $N$: total documents
- $df_j$: documents containing word

Chỉ số $tf-idf$ sẽ giúp đánh giá độ quan trọng của từ trong corpus và lọc bỏ những từ ít quan trọng như common words. $tf-idf$ càng lớn thì từ càng quan trọng

## 3. Model LDA (Latent Dirichlet Allocation)

- Assumptions of LDA for Topic Modeling
  - Documents with similar topics use similar groups of words
  - Latent topics can then be found by searching for groups of words that frequently occur together in documents across the corpus
    Model LDA là lớp mô hình sinh (generative model) cho phép xác định một tập hợp các chủ đề tương ứng

Let's get a high level overview of how LDA works for topic modeling

- Documents with similar topics use similar groups of words
- Latent topics can then be found by searching for groups of words that frequently occur together in documents across the corpus
- Documents are probability distributions over latent topics
- Topics themselves are probability distributions over words.

Các định nghĩa

Một số định nghĩa mà ta dùng sẽ sử dụng trong mô hinh LDA:

- Từ (word): là đơn vị cơ bản nhất của LDA. Một từ được xác định bởi một chỉ số index trong từ điển có giá trị từ $1,2,3,...V$. Một từ thứ $i$ được biểu diễn dưới dạng one-hot vector $w_i \in \mathbb{R}^V$ sao cho phần tử thứ $i$ của vector bằng 1 phần tử còn lại bằng 0.
- Văn bản (document): là tập hợp của $N$ từ được ký hiệu bởi $w = (w_1, w_2,...,w_N)$
- Bộ văn bản (corpus): là tập hợp của $M$ văn bản được ký kiệu bởi $D = w_1, w_2, w_3,... w_M$
- Topic ẩn (latent topic): là những chủ đề ẩn được xác định dựa trên phân phối của các từ và làm trung gian biển diễn các văn bản theo topic. Số lượng topic được xác định trước và ký hiệu $K$

## 4. Latent Dirichlet Allocation with Python

Ta sẽ làm một ví dụ về Latent Dirichlet Allocation với python

Import các thư viện cần thiết

```python
import pandas as pd
```

Đọc, và in 5 dòng đầu tệp dữ liệu

```python
npr = pd.read_csv('npr.csv')
npr.head()
```

Ta sẽ có kết quả như sau

```
Article
0 In the Washington of 2016, even when the polic...
1 Donald Trump has used Twitter — his prefe...
2 Donald Trump is unabashedly praising Russian...
3 Updated at 2:50 p. m. ET, Russian President Vl...
4 From photography, illustration and video, to d...
```

Biểu diễn văn bản thành mà trận từ

```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
dtm = cv.fit_transform(npr['Article'])
dtm
```

Áp dụng LDA

```python
from sklearn.decomposition import LatentDirichletAllocation
LDA = LatentDirichletAllocation(n_components=7, random_state=42)
LDA.fit(dtm)
```

Trong đó

- `LatentDirichletAllocation`: gọi thuật toán LDA trong sklearn
- `n_components=7`: giả sử có 7 topics cần rút ra từ tập văn bản
- `random_state=42`: cố định random seed để tái tạo kết quả
- `LDA.fit(dtm)`: huấn luyện mô hình trên ma trận từ `dtm`

```python
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

npr = pd.read_csv('/content/drive/MyDrive/AI ENGINEER/nlp course/Topic Modeling/npr.csv')
npr.head()
npr.info()

npr['Article'][0]

cv = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
dtm = cv.fit_transform(npr['Article'])
dtm

LDA = LatentDirichletAllocation(n_components=7, random_state=42)
LDA.fit(dtm)
```
