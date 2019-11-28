# EIP4-Assignment3

Final Validation accuracy for Base Network = 81.92

# Model Definition
model = Sequential()

model.add(SeparableConv2D(48, 3, 3, border_mode='same', input_shape=(32, 32, 3)))

model.add(Activation('relu'))

model.add(SeparableConv2D(48, 3, 3,use_bias=False)) #rf 3, nout = 30x30x48

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.1))


model.add(SeparableConv2D(96, 3, 3,use_bias=False)) #rf 5, nout = 28x28x96

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.1))


model.add(MaxPooling2D(pool_size=(2, 2))) #rf 6, nout = 14x14x96


model.add(SeparableConv2D(96,3,3,use_bias=False))  #rf 10, nout = 12x12x96

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.1))


model.add(SeparableConv2D(96,1,1,use_bias=False)) #rf 10, nout = 12x12x96

model.add(MaxPooling2D(pool_size=(2, 2))) #rf 12, nout = 6x6x96

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.1))




model.add(SeparableConv2D(192,3,3,use_bias=False)) #rf 20 , nout = 4x4x192

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.1))



model.add(SeparableConv2D(192,3,3,use_bias=False)) #rf 28 , nout = 2x2x192

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.1))


model.add(MaxPooling2D(pool_size=(2, 2))) #rf 32, nout = 1x1x192

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.1))


model.add(SeparableConv2D(10,1,1)) #rf 32, nout = 1x1x10


model.add(Flatten()) #10


model.add(Dense(256))

model.add(Activation('relu'))

model.add(Dropout(0.1))


model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#SeparableConv2D

# Epoch Logs

Highest val accuracy = 81.44

/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:12: UserWarning: The semantics of the Keras 2 argument `steps_per_epoch` is not the same as the Keras 1 argument `samples_per_epoch`. `steps_per_epoch` is the number of batches to draw from the generator at each epoch. Basically steps_per_epoch = samples_per_epoch/batch_size. Similarly `nb_val_samples`->`validation_steps` and `val_samples`->`steps` arguments have changed. Update your method calls accordingly.
  if sys.path[0] == '':
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:12: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<keras_pre..., validation_data=(array([[[..., verbose=1, steps_per_epoch=390, epochs=50)`
  if sys.path[0] == '':
Epoch 1/50
390/390 [==============================] - 34s 87ms/step - loss: 1.5463 - acc: 0.4234 - val_loss: 1.3109 - val_acc: 0.5168

Epoch 2/50
390/390 [==============================] - 25s 64ms/step - loss: 1.1663 - acc: 0.5767 - val_loss: 1.1337 - val_acc: 0.5950

Epoch 3/50
390/390 [==============================] - 25s 64ms/step - loss: 0.9874 - acc: 0.6473 - val_loss: 1.0742 - val_acc: 0.6209

Epoch 4/50
390/390 [==============================] - 25s 64ms/step - loss: 0.8812 - acc: 0.6891 - val_loss: 0.9378 - val_acc: 0.6682

Epoch 5/50
390/390 [==============================] - 25s 64ms/step - loss: 0.8094 - acc: 0.7128 - val_loss: 0.8736 - val_acc: 0.6963

Epoch 6/50
390/390 [==============================] - 25s 64ms/step - loss: 0.7439 - acc: 0.7379 - val_loss: 0.7969 - val_acc: 0.7254

Epoch 7/50
390/390 [==============================] - 25s 63ms/step - loss: 0.6983 - acc: 0.7550 - val_loss: 0.8044 - val_acc: 0.7192

Epoch 8/50
390/390 [==============================] - 25s 63ms/step - loss: 0.6653 - acc: 0.7642 - val_loss: 0.7167 - val_acc: 0.7499

Epoch 9/50
390/390 [==============================] - 25s 63ms/step - loss: 0.6353 - acc: 0.7772 - val_loss: 0.6908 - val_acc: 0.7625

Epoch 10/50
390/390 [==============================] - 25s 64ms/step - loss: 0.6068 - acc: 0.7868 - val_loss: 0.7121 - val_acc: 0.7508

Epoch 11/50
390/390 [==============================] - 25s 64ms/step - loss: 0.5814 - acc: 0.7966 - val_loss: 0.6955 - val_acc: 0.7539

Epoch 12/50
390/390 [==============================] - 25s 64ms/step - loss: 0.5670 - acc: 0.8016 - val_loss: 0.6998 - val_acc: 0.7610

Epoch 13/50
390/390 [==============================] - 25s 64ms/step - loss: 0.5408 - acc: 0.8089 - val_loss: 0.6649 - val_acc: 0.7744

Epoch 14/50
390/390 [==============================] - 25s 64ms/step - loss: 0.5277 - acc: 0.8132 - val_loss: 0.6406 - val_acc: 0.7780

Epoch 15/50
390/390 [==============================] - 25s 64ms/step - loss: 0.5122 - acc: 0.8202 - val_loss: 0.6223 - val_acc: 0.7889

Epoch 16/50
390/390 [==============================] - 25s 64ms/step - loss: 0.4982 - acc: 0.8244 - val_loss: 0.6535 - val_acc: 0.7737

Epoch 17/50
390/390 [==============================] - 25s 64ms/step - loss: 0.4830 - acc: 0.8302 - val_loss: 0.6152 - val_acc: 0.7907

Epoch 18/50
390/390 [==============================] - 25s 64ms/step - loss: 0.4713 - acc: 0.8354 - val_loss: 0.6761 - val_acc: 0.7690

Epoch 19/50
390/390 [==============================] - 25s 64ms/step - loss: 0.4626 - acc: 0.8372 - val_loss: 0.5949 - val_acc: 0.7981

Epoch 20/50
390/390 [==============================] - 25s 64ms/step - loss: 0.4575 - acc: 0.8373 - val_loss: 0.6386 - val_acc: 0.7839

Epoch 21/50
390/390 [==============================] - 25s 64ms/step - loss: 0.4389 - acc: 0.8449 - val_loss: 0.5892 - val_acc: 0.7936

Epoch 22/50
390/390 [==============================] - 25s 65ms/step - loss: 0.4333 - acc: 0.8471 - val_loss: 0.6368 - val_acc: 0.7828

Epoch 23/50
390/390 [==============================] - 25s 64ms/step - loss: 0.4257 - acc: 0.8490 - val_loss: 0.6281 - val_acc: 0.7880

Epoch 24/50
390/390 [==============================] - 25s 65ms/step - loss: 0.4125 - acc: 0.8533 - val_loss: 0.6111 - val_acc: 0.7926

Epoch 25/50
390/390 [==============================] - 25s 65ms/step - loss: 0.4063 - acc: 0.8563 - val_loss: 0.6198 - val_acc: 0.7895

Epoch 26/50
390/390 [==============================] - 25s 64ms/step - loss: 0.4011 - acc: 0.8577 - val_loss: 0.6190 - val_acc: 0.7883

Epoch 27/50
390/390 [==============================] - 25s 64ms/step - loss: 0.3953 - acc: 0.8591 - val_loss: 0.5936 - val_acc: 0.7983

Epoch 28/50
390/390 [==============================] - 25s 65ms/step - loss: 0.3925 - acc: 0.8617 - val_loss: 0.5599 - val_acc: 0.8085

Epoch 29/50
390/390 [==============================] - 25s 65ms/step - loss: 0.3872 - acc: 0.8609 - val_loss: 0.6088 - val_acc: 0.7923

Epoch 30/50
390/390 [==============================] - 25s 65ms/step - loss: 0.3764 - acc: 0.8662 - val_loss: 0.6206 - val_acc: 0.7898

Epoch 31/50
390/390 [==============================] - 25s 64ms/step - loss: 0.3661 - acc: 0.8693 - val_loss: 0.5890 - val_acc: 0.7983

Epoch 32/50
390/390 [==============================] - 25s 64ms/step - loss: 0.3638 - acc: 0.8696 - val_loss: 0.6098 - val_acc: 0.7963

Epoch 33/50
390/390 [==============================] - 25s 64ms/step - loss: 0.3611 - acc: 0.8704 - val_loss: 0.5803 - val_acc: 0.8074

Epoch 34/50
390/390 [==============================] - 25s 65ms/step - loss: 0.3556 - acc: 0.8715 - val_loss: 0.6530 - val_acc: 0.7810

Epoch 35/50
390/390 [==============================] - 25s 65ms/step - loss: 0.3510 - acc: 0.8750 - val_loss: 0.5854 - val_acc: 0.8040

Epoch 36/50
390/390 [==============================] - 25s 64ms/step - loss: 0.3469 - acc: 0.8763 - val_loss: 0.6104 - val_acc: 0.7995

Epoch 37/50
390/390 [==============================] - 25s 65ms/step - loss: 0.3392 - acc: 0.8777 - val_loss: 0.5977 - val_acc: 0.8046

Epoch 38/50
390/390 [==============================] - 25s 65ms/step - loss: 0.3390 - acc: 0.8783 - val_loss: 0.5747 - val_acc: 0.8118

Epoch 39/50
390/390 [==============================] - 25s 65ms/step - loss: 0.3344 - acc: 0.8815 - val_loss: 0.5658 - val_acc: 0.8122

Epoch 40/50
390/390 [==============================] - 25s 65ms/step - loss: 0.3364 - acc: 0.8802 - val_loss: 0.6456 - val_acc: 0.7911

Epoch 41/50
390/390 [==============================] - 25s 65ms/step - loss: 0.3285 - acc: 0.8839 - val_loss: 0.5697 - val_acc: 0.8144

Epoch 42/50
390/390 [==============================] - 25s 65ms/step - loss: 0.3199 - acc: 0.8860 - val_loss: 0.7090 - val_acc: 0.7796

Epoch 43/50
390/390 [==============================] - 25s 64ms/step - loss: 0.3199 - acc: 0.8861 - val_loss: 0.5913 - val_acc: 0.8072

Epoch 44/50
390/390 [==============================] - 25s 65ms/step - loss: 0.3153 - acc: 0.8873 - val_loss: 0.5810 - val_acc: 0.8100

Epoch 45/50
390/390 [==============================] - 25s 65ms/step - loss: 0.3143 - acc: 0.8878 - val_loss: 0.6057 - val_acc: 0.8051

Epoch 46/50
390/390 [==============================] - 25s 65ms/step - loss: 0.3048 - acc: 0.8908 - val_loss: 0.6306 - val_acc: 0.8000

Epoch 47/50
390/390 [==============================] - 25s 65ms/step - loss: 0.3064 - acc: 0.8899 - val_loss: 0.5882 - val_acc: 0.8070

Epoch 48/50
390/390 [==============================] - 25s 65ms/step - loss: 0.3013 - acc: 0.8910 - val_loss: 0.6064 - val_acc: 0.8087

Epoch 49/50
390/390 [==============================] - 25s 65ms/step - loss: 0.2983 - acc: 0.8924 - val_loss: 0.5774 - val_acc: 0.8096

Epoch 50/50
390/390 [==============================] - 25s 65ms/step - loss: 0.2980 - acc: 0.8927 - val_loss: 0.7008 - val_acc: 0.7813

Model took 1265.26 seconds to train

Accuracy on test data is: 78.13

#### Our team members are Hiranmai Tummalapalli and Raviteja Penugonda
