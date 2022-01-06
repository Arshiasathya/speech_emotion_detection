import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import keras

from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.utils import np_utils

import data_feature_visualization, label_tagging, data_split, metrics, model_loading


class Audio_model:
    """
    Audio signal classification for speec emotion detection, this model uses Keras CNN classifier to detect the emotion
    of the speech, this project is build based on the following data set
    1)  Toronto emotional speech set (TESS)
    2) Ryerson Audio-Visual Database of Emotional

    Work in progress - trying to make this work on a real time detection also looking for more data samples
    """
    def feature_extraction(self, my_audio_list, colum_name, labels):
        """
        :param my_audio_list: List - list of audio files
        :param colum_name: string - name of the feature from the audio file
        :param labels: string - Emotions e.g. Happy, Neutral, Sad
        :return: Dataframes - features dataframe and labels dataframe
        """
        df = pd.DataFrame(columns=[colum_name])
        bookmark = 0
        for index, y in enumerate(my_audio_list):
            if my_audio_list[index][6:-16] != '01' and my_audio_list[index][6:-16] != '07' and my_audio_list[index][
                                                                                               6:-16] != '08':
                X, sample_rate = librosa.load('/home/dev-ml/Documents/Notebooks/Audio_Speech_Actors_01-24/' + y,
                                              res_type='kaiser_fast', duration=3, sr=22050 * 2, offset=0.5)
                sample_rate = np.array(sample_rate)
                mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=12), axis=0)
                feature = mfccs
                df.loc[bookmark] = [feature]
                bookmark = bookmark + 1
        df1 = pd.DataFrame(df[colum_name].values.tolist())
        features_df = pd.concat([df1, labels], axis=1)
        features = features_df.rename(index=str, columns={"0": "labels"})

        return features

    def label_encoding(self, train, test):
        """
        :param train: train dataframe
        :param test: test dataframe
        :return:encoded train test label dataframes with label encoder for future predictions
        """
        trainfeatures = train.iloc[:, :-1]

        trainlabel = train.iloc[:, -1:]
        testfeatures = test.iloc[:, :-1]
        testlabel = test.iloc[:, -1:]

        X_train = np.array(trainfeatures)
        y_train = np.array(trainlabel)
        X_test = np.array(testfeatures)
        y_test = np.array(testlabel)

        lb = LabelEncoder()

        y_train = np_utils.to_categorical(lb.fit_transform(y_train))
        y_test = np_utils.to_categorical(lb.fit_transform(y_test))

        return X_train, y_train, X_test, y_test, lb

    def expand_dims(self, df):
        """
        :param df: dataframe with 2d features
        :return: df - expanded to support CNN
        """
        return np.expand_dims(df, axis=2)

    def build_model(self):
        """
        :return: Sequential Keras model
        """
        model = Sequential()

        model.add(Conv1D(128, 5, padding='same',
                         input_shape=(216, 1)))
        model.add(Activation('relu'))
        model.add(Conv1D(128, 5, padding='same'))
        model.add(Activation('relu'))
        model.add(Dropout(0.1))
        model.add(MaxPooling1D(pool_size=(8)))
        model.add(Conv1D(128, 5, padding='same', ))
        model.add(Activation('relu'))
        model.add(Conv1D(128, 5, padding='same', ))
        model.add(Activation('relu'))
        model.add(Conv1D(128, 5, padding='same', ))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Conv1D(128, 5, padding='same', ))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(10))
        model.add(Activation('softmax'))
        opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)

        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        return model


    def fit(self, model, X_train, y_train, batch_size, epoch, X_test, y_test):
        """
        :param model: Sequential model
        :param X_train: train features dataframe
        :param y_train: train labels
        :param batch_size: int batch size
        :param epoch: int number of epochs for training
        :param X_test: test features dataframe
        :param y_test: test labels
        :return: trained keras model
        """
        trained_model = model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch, validation_data=(X_test, y_test))
        return trained_model


    def save_model(self, model_name, save_dir, model):
        """
        :param model_name: string name of the model to be saved
        :param save_dir: string where to save the model
        :param model: h5 file trained keras model
        :return: save the model in the given location
        """
        # save model and weights
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, model_name)
        model.save(model_path)
        print('Save trained model at %s' % model_path)


    def testing(self, loaded_model, X_test,lb):
        """
        :param loaded_model: saved model h5 file
        :param X_test: test dataframe
        :param lb : saved label encoder
        :return: predicted dataframe with predicted labels
        """
        preds = loaded_model.predict(X_test, batch_size=4, verbose=1)
        pred1 = preds.argmax(axis=1)
        label_name = pred1.astype(int).flatten()
        predictions = (lb.inverse_transform(label_name))
        prediction_df = pd.DataFrame({'prediction_values': predictions})

        actual = y_test.argmax(axis=1)
        actual_label_name = actual.astype(int).flatten()
        actual_values = (lb.inverse_transform((actual_label_name)))
        actual_values_df = pd.DataFrame({'actual_values': actual_values})

        predicted_actual_df = actual_values_df.join(prediction_df)
        print(predicted_actual_df[170:180])
        return predicted_actual_df


    def load_and_preprocess_data(self, audio_list):
        """
        :param audio_list: list of audio files
        :return: features dataframe and labels dataframe
        """
        data_feature_visualization.visualize_data(audio_list[1])
        data_feature_visualization.visualize_feature(audio_list[1])
        labels_df = label_tagging.add_labels(audio_list)
        audio_features = audio_model.feature_extraction(audio_list, 'MFCC_feature', labels_df)
        return labels_df, audio_features

    def split_data(self, audio_features):
        """
        :param audio_features: full set of features dataframe
        :return: train dataframe, test dataframe, train labels, test labels
        """
        train_df, test_df = data_split.train_test_split(audio_features)
        X_train, y_train, X_test, y_test, lb = audio_model.label_encoding(train_df, test_df)
        X_train = audio_model.expand_dims(X_train)
        X_test = audio_model.expand_dims(X_test)

        return X_train, y_train, X_test, y_test
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    audio_model = Audio_model()
    audio_list = 'location_of_where the files stored'
    labels_df, audio_features = audio_model.load_and_preprocess_data(audio_list)
    X_train, y_train, X_test, y_test = audio_model.split_data(audio_features)
    model = audio_model.build_model()
    trained_audio_model = audio_model.fit(model, X_train, y_train, 32, 300, X_test, y_test)
    metrics.plot_loss(trained_audio_model)
    metrics.accuracy_plot(trained_audio_model)
    model_name = 'Speech_Emotion_Detecor_to_demo_50.h5'
    save_dir = os.path.join(os.getcwd(), 'saved_models')  #path where the model will be saved
    audio_model.save_model(model_name, save_dir, trained_audio_model)
    loaded_model = model_loading.load_model(save_dir, model_name)
    score = loaded_model.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
