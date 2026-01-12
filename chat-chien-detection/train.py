import os
import numpy as np
import matplotlib.pyplot as plt
from model import create_model

DATA_DIR = 'data'
MODEL_PATH = 'cat_dog_model.h5'
EPOCHS = 10
BATCH_SIZE = 32

def load_data():
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
    return X_train, X_test, y_train, y_test

def train_model():
    X_train, X_test, y_train, y_test = load_data()
    model = create_model()
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test)
    )
    
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Précision : {test_acc*100:.2f}%")
    
    model.save(MODEL_PATH)
    print(f"Modèle sauvegardé!")
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Précision')
    plt.legend(['Train', 'Test'])
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Perte')
    plt.legend(['Train', 'Test'])
    plt.savefig('training.png')
    plt.show()

if __name__ == '__main__':
    train_model()