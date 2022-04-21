from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from utils.charts import writeLogFile, trainValidationPlot
from datetime import datetime
import tensorflow as tf
import time


def main():
    startTime = time.time()

    # ------ setting up the model --------
    classifier_vgg16 = VGG16(input_shape=(64, 64, 3),
                             include_top=False,
                             weights="imagenet")

    for layer in classifier_vgg16.layers:
        layer.trainable = False

    # ------ adding extra layers ------
    model = Sequential()
    model.add(classifier_vgg16)
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(32, kernel_initializer="he_uniform"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(32, kernel_initializer="he_uniform"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(32, kernel_initializer="he_uniform"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dense(7, activation="softmax"))

    model.summary()

    # --------------- data augmentation ---------
    train_data_generator = ImageDataGenerator(rescale=1. / 255,
                                              rotation_range=5,
                                              validation_split=0.2,
                                              width_shift_range=0.2,
                                              height_shift_range=0.2,
                                              vertical_flip=True,
                                              horizontal_flip=True,
                                              fill_mode="nearest")

    valid_data_generator = ImageDataGenerator(rescale=1. / 255,
                                              validation_split=0.2)

    # ---- loading data ------
    training_set_path = "database/FER-2013/train"
    training_set = train_data_generator.flow_from_directory(training_set_path,
                                                            target_size=(64, 64),
                                                            batch_size=64,
                                                            class_mode="categorical",
                                                            subset="training")

    validation_set = valid_data_generator.flow_from_directory(training_set_path,
                                                              target_size=(64, 64),
                                                              batch_size=64,
                                                              class_mode="categorical",
                                                              subset="validation")

    # -------- metrics for the compilation -----
    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]

    # ------ callbacks -----
    lr_reduce = ReduceLROnPlateau(monitor="val_loss", patience=20, verbose=1, factor=0.50, min_lr=1e-10)
    model_checkpoint = ModelCheckpoint("models/model.h5")
    early_stopping = EarlyStopping(verbose=1, patience=20)

    # ------ compilation of the model ------
    model.compile(optimizer="Adam",
                  loss="categorical_crossentropy",
                  metrics=METRICS)

    # ----------- fitting the model -------
    callbacks = [lr_reduce, model_checkpoint, early_stopping]
    history = model.fit(training_set,
                        validation_data=validation_set,
                        epochs=1,
                        steps_per_epoch=3,
                        verbose=1,
                        callbacks=callbacks)

    # ----- writing the log files ------
    finalAccuracy = int(float(history.history["accuracy"][0]) * 1000)
    date = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    filename = f"vgg16_{finalAccuracy}_{date}"

    # ------ saving details about training -----
    writeLogFile(date, startTime, time.time(),
                 history.history["accuracy"],
                 history.history["loss"],
                 history.history["auc"],
                 history.history["precision"],
                 history.history["val_accuracy"],
                 history.history["val_loss"],
                 history.history["val_auc"],
                 history.history["val_precision"],
                 filename)

    # ----- saving the model as a .hdf5 file -----
    model.save(f"models/{filename}/{filename}.hdf5")

    # ------- creating histograms -------
    trainValidationPlot(history.history["accuracy"], history.history["val_accuracy"],
                        history.history["loss"], history.history["val_loss"],
                        history.history["auc"], history.history["val_auc"],
                        history.history["precision"], history.history["val_precision"], filename)


if __name__ == "__main__":
    main()
