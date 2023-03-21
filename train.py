import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers
from keras import models
from keras_preprocessing.image import img_to_array
from keras_preprocessing.image import load_img


epochs = 10       #訓練的次數
img_rows = None   #驗證碼影像檔的高
img_cols = None   #驗證碼影像檔的寬
digits_in_img = 4 #驗證碼碼數
x_list = list()   #存所有驗證碼的list
y_list = list()   #存所有驗證碼的list代表的正確字元
x_train = list()  #存訓練用驗證碼的list
y_train = list()  #存訓練用驗證碼list代表的正確字元
x_test = list()   #存測試用驗證碼的list
y_test = list()   #存測試用驗證碼list代表的正確字元

def split_digits_in_img(img_array, x_list, y_list):
    for i in range(digits_in_img):
        step = img_cols // digits_in_img
        x_list.append(img_array[:, i * step:(i + 1) * step] / 255)
        y_list.append(img_filename[i])

img_filenames = os.listdir('captcha_img')

for img_filename in img_filenames:
    if '.png' not in img_filename:
        continue
    img = load_img('captcha_img/{0}'.format(img_filename), color_mode='grayscale')
    img_array = img_to_array(img)
    img_rows, img_cols, _ = img_array.shape
    split_digits_in_img(img_array, x_list, y_list)

y_list = [ord(y)-97 for y in y_list]
y_list = keras.utils.to_categorical(y_list, num_classes=26)
x_train, x_test, y_train, y_test = train_test_split(x_list, y_list)

if os.path.isfile('cnn_model.h5'):
    model = models.load_model('cnn_model.h5')
    print('Model loaded from file.')
else:
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols // digits_in_img, 1)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(rate=0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(26, activation='softmax'))
    print('New model created.')
 
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

model.fit(np.array(x_train), np.array(y_train), batch_size=100, epochs=epochs, verbose=1, validation_data=(np.array(x_test), np.array(y_test)))
 
loss, accuracy = model.evaluate(np.array(x_test), np.array(y_test), verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)
 
model.save('cnn_model.h5')