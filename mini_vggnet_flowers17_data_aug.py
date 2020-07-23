from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing_tools.image_to_array_preprocessor import ImageToArrayPreprocessor
from preprocessing_tools.aspectawarepreprocessor import AspectAwarePreprocessor
from preprocessing_tools.datasetloader import SimpleDatasetLoader
from preprocessing_tools.mini_vggnet import MiniVGGNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

print("[INFO] loading data...")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [imagePath.split(os.path.sep)[-2] for imagePath in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

aap = AspectAwarePreprocessor(width=64, height=64)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float")/255.0

(X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size=0.25)

lb = LabelBinarizer().fit(y_train)
y_train = lb.transform(y_train)
y_test = lb.transform(y_test)

# this gives a generator
aug = ImageDataGenerator(rotation_range = 30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

print("[INFO] compiling model...")
model = MiniVGGNet.build(height=64, width=64, depth=3, classes=len(classNames))
opt = SGD(lr=0.05)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit_generator(aug.flow(X_train, y_train, batch_size=32), validation_data=(X_test, y_test), steps_per_epoch = len(X_train)//32, epochs=100, verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(X_test, batch_size=32)
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()


