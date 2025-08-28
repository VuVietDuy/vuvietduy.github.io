---
layout: post
title: "Nhận diện hoa (flower recognition) bằng CNN"
author: vuvietduy
categories: [Machine Learning, Tutorial]
featured: false
published: true
image:
toc: true
excerpt: ""
---

## Giới thiệu mạng CNN

CNN (Convolutional Neural Network) là một mạng nơ-ron nhân tạo trong lĩnh vực computer vision

## Kiến trúc của CNN

1. Input layer

Nhận dữ liệu đầu vào, thường là anh dưới dạng ma trận 3 chiều

2. Convolutional Layer

Dùng kernel/filter để quét trên anh và tạo feature maps. Giúp trích xuất đặc trưng

_feature map (bản đồ đặc trưng): là ma trận đầu ra khi áp dụng phép tích chập giữa bộ lọc với các vung nhận thức theo phương di chuyển từ trái qua phải, từ trên xuống dưới_

3. Activation Layer

Thường dùng ReLU, giúp mạng học được đặc trưng phi tuyến

$f(x) = max(0, x)$

4. Pooling Layer

Thường được dùng giữa các convolutional layer, để giảm kích thước dữ liệu nhưng vẫn giữ được các thuộc tính quan trọng. Kích thước dữ liệu giảm giúp giảm việc tính toán trong model

5. Dropout Layer (tùy chọn)

Bỏ ngẫu nhiên một số neuron trong lúc training để tráng overfiting

6. Fully Connected Layer (Dens Layer)

Sau khi ảnh được truyền qua nhiều convolutional layer và pooling layer thì model đã học được tương đối đặc điểm của ảnh thì tensor của output của layer cuối cùng, kích thước H*W*D, sẽ được chuyển về 1 vector kích thước H*W*D

Nối tất cả các neuron từ layer trước (sau khi flatten feature map). Giúp học mối quan hệ giữa đặc trưng và nhãn. Cuối cùng dùng softmax (cho phân loại nhiều lớp) hoặc sigmoid (cho nhị phân)

6. Output Layer

Kết quả dự đoán, thường là:

- Softmax &rarr; vector xác suất cho từng lớp.
- Sigmoid &rarr; xác suất một lớp (binary classification).

## Thực hành xây dựng mô hình CNN cho bài toán nhận diện hoa

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_dir = 'flowers/'
img_size = 224
batch = 64

datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    base_dir + 'train',
    target_size=(img_size, img_size),
    batch_size=batch,
    class_mode='categorical',
    subset="training"
)

val_generator = datagen.flow_from_directory(
    base_dir + 'train',
    target_size=(img_size, img_size),
    batch_size=batch,
    class_mode='categorical',
    subset="validation"
)
```

Khởi tạo model Sequential, tức là mạng sẽ được xây dựng theo dạng xếp chồng tầng (layer) nối tiếp nhau

Thêm lớp tích chập 2D: Conv2D

- `filters=64`: có 64 bộ lọc &rarr; trích xuất được 64 đặc trưng khác nhau từ ảnh
- `kernel_size=(5,5)`: mỗi filter là ma trận 5×5.
- `padding='same'`: giữ nguyên kích thước ảnh đầu ra (padding thêm pixel).
- `activation='relu'`: áp dụng hàm kích hoạt ReLU để đưa các giá trị âm về 0 &rarr; giúp học đặc trưng phi tuyến.
- `input_shape=(224,224,3)`: kích thước ảnh đầu vào là 224×224 RGB (3 kênh màu).

Lớp Pooling giảm kích thước đặc trưng (feature map), chỉ lấy giá trị lớn nhất trong mỗi ô 2×2 &rarr; giảm số tham số, tránh overfitting.

Biến đổi feature map 2D &rarr; vector 1D để đưa vào các tầng fully connected.

Dense layer với 512 neuron, kết hợp toàn bộ đặc trưng lại để học mối quan hệ.
Dùng ReLU để giữ phi tuyến tính.

Lớp đầu ra (output layer):

Dense(5): có 5 neuron, tương ứng với 5 lớp (categories) trong bài toán phân loại.

softmax: biến vector thành xác suất &rarr; lớp nào có xác suất cao nhất thì đó là dự đoán của mô hình.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(5, activation="softmax"))
```

Ta sẽ có được mô hình với các tham số như sau

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d_8 (Conv2D)               │ (None, 224, 224, 64)   │         4,864 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_5 (MaxPooling2D)  │ (None, 112, 112, 64)   │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_9 (Conv2D)               │ (None, 112, 112, 64)   │        36,928 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_6 (MaxPooling2D)  │ (None, 56, 56, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_10 (Conv2D)              │ (None, 56, 56, 64)     │        36,928 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_7 (MaxPooling2D)  │ (None, 28, 28, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_11 (Conv2D)              │ (None, 28, 28, 64)     │        36,928 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_8 (MaxPooling2D)  │ (None, 14, 14, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 12544)          │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 512)            │     6,423,040 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ activation (Activation)         │ (None, 512)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 5)              │         2,565 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
```

```python
import tensorflow as tf

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy', metrics=['accuracy'])
```

```python
epochs = 30
model.fit(
    train_generator, epochs=epochs, validation_data=val_generator
)
model.save('model.h5')
```

```python
from tensorflow.keras.models import load_model
savedModel=load_model('model.h5')
```

```python
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

list_ = ['Daisy','Danelion','Rose','sunflower', 'tulip']
test_image = image.load_img('Image_1.jpg',target_size=(224,224))

plt.imshow(test_image)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)

result = savedModel.predict(test_image)
print(result)
i=0
for i in range(len(result[0])):
  if(result[0][i]==1):
    print(list_[i])
    break
```
