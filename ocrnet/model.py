import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Flatten, Dense

class OCRModel(tf.keras.Model):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.num_classes = num_classes

        self.conv1 = Conv2D(16, (3,3), padding="same",activation=tf.nn.relu)
        self.pool1 = MaxPooling2D(strides=2, padding="same")
        
        self.conv2 = Conv2D(32, (3,3), padding="same" ,activation=tf.nn.relu)
        self.pool2 = MaxPooling2D(strides=2, padding="same")
        
        self.conv3 = Conv2D(64, (3,3), padding="same" ,activation=tf.nn.relu)
        self.conv4 = Conv2D(64, (3,3), padding="same" ,activation=tf.nn.relu)
        self.pool3 = MaxPooling2D((1,2),strides=2, padding="same")
        
        self.conv5 = Conv2D(64, (3,3), padding="same")
        self.bnorm1 = BatchNormalization(3)
        self.relu1 = Activation("relu")
        
        self.conv6 = Conv2D(64, (2,2) ,activation=tf.nn.relu)
        self.flatten = Flatten()
        
        self.dense1 = Dense(128, activation='relu')
        self.classifier = Dense(num_classes, activation='softmax')
    
    def call(self, inputs):
        x = self.pool1(self.conv1(inputs))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv4(self.conv3(x)))
        x = self.relu1(self.bnorm1(self.conv5(x)))
        x = self.flatten(self.conv6(x))
        return self.classifier(self.dense1(x))


def build_model(classes):
    model = OCRModel(classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                     metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model

