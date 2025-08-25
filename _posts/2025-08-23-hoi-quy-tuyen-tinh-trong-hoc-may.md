---
layout: post
title: "Hồi quy tuyến tính trong học máy"
author: vuvietduy
categories: ["Học máy", Tutorial]
featured: false
published: false
image:
toc: true
excerpt: "Hồi quy tuyến tính trong học máy"
---

Bài toán: Xét bài toán ước lượng giá của một căn nhà rộng $x_1$ $m^2$, có $x_2$ phòng ngủ và cách trung tâm thành phố $x_3$ km. Giả sử có một tập dữ liệu của 1000 căn nhà trong thành phố đó. Liệu rằng khi có một căn nhà mới với các thông số về diện tích $x_1$, số phòng ngủ $x_2$, cách trung tâm thành phố $x_3$, chúng ta có thể dự đoán được giá $y$ của căn nhà đó không?

Nếu có thì hàm dự đoán $y = f(x)$ có dạng như nào?

Ở đây vector đặc trưng $x = [x_1, x_2, x_3]^T$ là một vector cột chứa dữ liệu đầu vào, đầu ra là một số thực dương $y$

## Xây dựng hàm mất mát

Nếu mỗi điểm dữ liệu được mô tả bởi 1 vector đặc trưng $d$ chiều, $x \in \mathbb{R}^d$, hàm dự đoán đầu ra được viết dưới dạng

$y = w_1x_1 + w_2x_2 + ... + w_dx_d = x_TW$

## Sai số dự đoán

Sau khi xây dựng được mô hình sự đoán đầu ra, ta cần đánh giá sai số

$\frac{1}{2}e^2 = \frac{1}{2}(y - y)^2$

## Sai số huấn luyện
