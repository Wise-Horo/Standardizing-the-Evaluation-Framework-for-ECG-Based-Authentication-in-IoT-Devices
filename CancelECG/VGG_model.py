from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Activation, Input

feature_extractor = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in feature_extractor.layers:
    layer.trainable = False
x = feature_extractor.output
x = Flatten()(x)
x = Dense(4096, activation='relu')(x)  # fc1
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu')(x)  # fc2
x = Dropout(0.5)(x)
x = Dense(1000, activation='relu')(x)  # fc3 (output)

model = Model(inputs=feature_extractor.input, outputs=x)
model.summary()
# Extract deep features for all images
for layer in model.layers:
    layer.trainable = False
