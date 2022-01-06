import matplotlib.pyplot as plt
"""
These function to support tracking the model metrics, future need to add confusion matrix
"""
def plot_loss(model):
    """This plots the loss of the trained model"""
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()


def accuracy_plot(model):
    """To plot the accuracy of the trained on the validation or test set"""
    plt.plot(model.history['acc'])
    plt.plot(model.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()

