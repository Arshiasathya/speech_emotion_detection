from keras.models import model_from_json
import keras


def load_model(saved_dir, model):
    """
    :param saved_dir: srting where the model is saved
    :param model: model to be loaded
    :return: load the model for prediction
    """
    # Loading json and creating model
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    # loaded_model.load_weights("/home/dev-ml/Documents/Notebooks/saved_models/Speech_Emotion_Detecor_to_demo_300.h5")
    # loaded_model.load_weights("/home/dev-ml/Documents/Notebooks/saved_models/Speech_Emotion_Detecor_to_demo_50_1.h5")
    loaded_model.load_weights("/home/dev-ml/Documents/Notebooks/saved_models/Speech_Emotion_Detecor_to_demo_50.h5")
    print("Loaded model from disk")
    opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)

    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return loaded_model