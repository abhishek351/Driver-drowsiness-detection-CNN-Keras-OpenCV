from keras.models import Sequential
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten
from keras.preprocessing.image import ImageDataGenerator


train = ImageDataGenerator(rescale=1/255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
validation = ImageDataGenerator(rescale=1/255)

train_image_gen = train.flow_from_directory('Train dataset',target_size=(100,100),batch_size=32,color_mode='grayscale',class_mode='binary')
valid_image_gen = validation.flow_from_directory('Valid dataset',target_size=(100,100),batch_size=32,color_mode='grayscale',class_mode='binary')

batch_size=32
SPE= len(train_image_gen.classes)//batch_size 
VS = len(valid_image_gen.classes)//batch_size 
print(SPE,VS)

print(train_image_gen.class_indices)

model = Sequential()
model.add(Conv2D(64,(3,3),input_shape=(100,100,1),activation="relu"))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(32,(3,3),activation="relu"))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(32,(3,3),activation="relu"))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(16,(3,3),activation="relu"))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(16,(3,3),activation="relu"))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(128,activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(64,activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

results = model.fit_generator(train_image_gen,validation_data=valid_image_gen,epochs=10,steps_per_epoch=SPE,validation_steps=VS)

model.save("CNN__model.h5")