# -*- coding: utf-8 -*-
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import json
import matplotlib.pyplot as plt

# seed 고정
seed = 15
np.random.seed(seed)
tf.random.set_seed(seed)

# 설정값
BATCH_SIZE = 16
EPOCHS = 50
IMG_HEIGHT = 224
IMG_WIDTH = 224

# 🔹 로컬 데이터셋 경로 직접 지정
# 예: dataset 폴더 안에 train / validation 디렉토리 존재해야 함
base_dir = "C:/Users/itg/Pictures/wheat"   # 👉 여기를 본인 PC 경로로 바꿔주세요
train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")

# 1️⃣ image_dataset_from_directory로 데이터 로드
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=False
)

# 2️⃣ 데이터 증강 함수
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.05)
    return image, label

# 3️⃣ 비율 유지 + 패딩 + 전처리
def resize_pad_preprocess(image, label):
    image = tf.image.resize_with_pad(image, IMG_HEIGHT, IMG_WIDTH)
    image = preprocess_input(image)
    return image, label

# 클래스 이름 저장용
class_list = train_ds.class_names

# 4️⃣ 파이프라인 구성
AUTOTUNE = tf.data.AUTOTUNE
train_ds = (train_ds
            .map(augment, num_parallel_calls=AUTOTUNE)
            .map(resize_pad_preprocess, num_parallel_calls=AUTOTUNE)
            .apply(tf.data.experimental.ignore_errors())
            .cache()
            .prefetch(AUTOTUNE))

val_ds = (val_ds
          .map(resize_pad_preprocess, num_parallel_calls=AUTOTUNE)
          .apply(tf.data.experimental.ignore_errors())
          .cache()
          .prefetch(AUTOTUNE))

# 5️⃣ DenseNet121 불러오기
base_model = DenseNet121(weights="imagenet", include_top=False,
                         input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# 전이학습: 기본 가중치는 freeze
for layer in base_model.layers:
    layer.trainable = False

# 6️⃣ 출력층 추가
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.5)(x)
output_layer = Dense(len(class_list), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output_layer)
model.summary()

# 7️⃣ 컴파일
loss_fn = CategoricalCrossentropy(label_smoothing=0.1)
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=loss_fn,
    metrics=["accuracy"]
)

# 8️⃣ 학습
early_stop = EarlyStopping(monitor="val_loss", patience=5,
                           restore_best_weights=True)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[early_stop]
)

# 9️⃣ 학습 결과 그래프
plt.plot(history.history["accuracy"], label="train_accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 🔟 모델 + 클래스 저장 (로컬 PC에 저장)
save_dir = "models"  # 👉 원하는 저장 경로로 변경
os.makedirs(save_dir, exist_ok=True)

save_model_path = os.path.join(save_dir, "Densenet_Xray.h5")
save_label_path = os.path.join(save_dir, "Densenet_Xray.json")

model.save(save_model_path)
with open(save_label_path, "w") as f:
    json.dump(class_list, f)

print(f"✅ 모델 저장 완료: {save_model_path}")
print(f"✅ 클래스 저장 완료: {save_label_path}")
