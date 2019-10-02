#train_pipeline
import utilsT
from sklearn.model_selection import train_test_split
import keras

def main():
    image_size = 28
    frame_duration = 1.0
    overlap = 0.5 # Pipeline 3, 4: 0.75; Pipeline: 5: 0

    X, y = utilsT.make_data_pipeline(file_names,labels,image_size,frame_duration,overlap)
    # X.shape: 3642, 28, 28, 3
    # y.shape: 3642

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True)

    # input image dimensions
    img_rows, img_cols = 28, 28

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    input_shape = (img_rows, img_cols, 3)

    batch_size = 128 # Rev 0: 32
    num_classes = 2
    epochs = 2 # 400 # Pipeline 2, 3: epochs = 500 # Rev 0: 200

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = models.get_model_1(input_shape, num_classes)
    # model = models.get_model_2(input_shape, num_classes)

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6) # Rev: lr=0.0001

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    #x_train /= 255
    #x_test /= 255


    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)

if __name__ == "__main__":
    main()
