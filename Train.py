
def training():
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Conv2D
    from keras.layers import MaxPooling2D
    from keras.layers import Dense
    from keras.layers import Flatten
    from keras.optimizers import SGD
    from keras.utils import to_categorical
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    train_norm = X_train.astype('float32')
    test_norm = X_test.astype('float32')
    X_train = train_norm / 255.0
    X_test= test_norm / 255.0


    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    _, acc = model.evaluate(X_test, y_test)
    print('%.3f' % (acc * 100.0))

    model.save('modelx.h5')

if __name__ == "__main__":
    training()