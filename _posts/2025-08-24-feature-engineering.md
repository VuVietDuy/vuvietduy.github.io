---
layout: post
title: "Kỹ thuật feature engineering"
author: vuvietduy
categories: [Blog, Tutorial]
featured: false
published: true
image:
toc: true
excerpt: "Hướng dẫn sử dụng markdown"
---

## Trích lọc đặc trưng

Mã hóa: chia đoạn văn thành các câu văn, các câu văn thành các từ. Trong mã hóa, từ là đơn vị cơ sở. Cần một bộ tokenizer có kích thước bằng toàn bộ các từ xuất hiện trong văn bản hoặc bằng toàn bộ các từ có trong từ điển. Một câu văn sẽ được biểu diễn bằng một sparse vector mà mỗi một phần tử đại điện cho một từ. Các tokemizer sẽ khác nhau cho mỗi một ngôn ngữ khác nhau.
