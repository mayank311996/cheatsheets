from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, \
    GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from scipy.spatial import distance as dist
from sklearn.metrics.pairwise import cosine_distances

# For ResNet and VGG16
# from tensorflow.keras.applications.resnet50 import ResNet50, \
#    preprocess_input
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.applications.vgg16 import preprocess_input

##############################################################################
# ResNet and VGG16, img_width, img_height = 224, 224
img_width, img_height = 299, 299

NUM_EPOCHS = 5
batchsize = 10
num_train_images = 900
num_val_images = 100

base_model = InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(img_height, img_width, 3)
)

##############################################################################
train_dir = 'train_dir'
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True
)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batchsize
)

val_dir = 'val_dir'
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batchsize
)


##############################################################################
def build_final_model(base_model, dropout, fc_layers, num_classes):
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)

    for fc in fc_layers:
        x = Dense(fc, activation='relu')(x)
        x = Dropout(dropout)(x)

    predictions = Dense(num_classes, activation='softmax')(x)
    final_model = Model(
        inputs=base_model.input,
        outputs=predictions
    )

    return final_model


##############################################################################
class_list = ["bed", "chair", "sofa"]
FC_LAYERS = [1024, 1024]
dropout = 0.3

final_model = build_final_model(
    base_model,
    dropout=dropout,
    fc_layers=FC_LAYERS,
    num_classes=len(class_list)
)

adam = Adam(lr=0.00001)
final_model.compile(
    adam,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
history = final_model.fit_generator(
    train_generator,
    epochs=NUM_EPOCHS,
    workers=0,
    steps_per_epoch=num_train_images//batchsize,
    shuffle=True,
    validation_data=val_generator,
    validation_steps=num_val_images//batchsize
)

##############################################################################
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 5.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

##############################################################################
# the below path shows the full path for uploaded image, adjust it for your
# specific path
img_path = 'furniture_images/train/bed/bed00000002train.jpg'
# img_path ='/home/krish/visual_search/furniture-detector/img/train/chair
# /00000009.jpg'
# img_path ='/home/krish/visual_search/furniture-detector/img/train/sofa/
# 00000016.jpg'
img = image.load_img(img_path, target_size=(img_width, img_height))
img_data = image.img_to_array(img)
img_data = np.expand_dims(img_data, axis=0)
img_data1 = preprocess_input(img_data)

pretrained_feature = final_model.predict(img_data, verbose=0)
pretrained_feature_np = np.array(pretrained_feature)
pretrained_feature1D = pretrained_feature_np.flatten()

pretrained_feature_base = base_model.predict(img_data1)
pretrained_feature_np_base = np.array(pretrained_feature_base)
pretrained_feature1D_base = pretrained_feature_np_base.flatten()


print(pretrained_feature1D)
y_prob = final_model.predict(img_data)

y_classes = y_prob.argmax(axis=-1)

print(y_classes)

##############################################################################
# Below shows the full path for test images, adjust it for your specific path
if y_classes == [0]:
    path = 'furniture_images/val/bed'
elif y_classes == [1]:
    path = 'furniture_images/val/chair'
else:
    path = 'furniture_images/val/sofa'

mindist = 10000
maxcosine = 0
i = 0
for filename in os.listdir(path):
    image_train = os.path.join(path, filename)
    i += 1
    imgtrain = image.load_img(
        image_train,
        target_size=(img_width, img_height)
    )
    img_data_train = image.img_to_array(imgtrain)
    img_data_train = np.expand_dims(img_data_train, axis=0)
    img_data_train = preprocess_input(img_data_train)

    pretrained_feature_train = base_model.predict(img_data_train)
    pretrained_feature_np_train = np.array(pretrained_feature_train)
    pretrained_feature_train1D = pretrained_feature_np_train.flatten()
    eucldist = dist.euclidean(
        pretrained_feature1D_base,
        pretrained_feature_train1D)

    if mindist > eucldist:
        mindist = eucldist
        minfilename = filename
    # print (vgg16_feature_np)

    dot_product = np.dot(pretrained_feature1D_base,
                         pretrained_feature_train1D)
    # normalize the results, to achieve similarity measures independant
    # of the scale of the vectors
    norm_Y = np.linalg.norm(pretrained_feature1D_base)
    norm_X = np.linalg.norm(pretrained_feature_train1D)
    cosine_similarity = dot_product / (norm_X * norm_Y)

    if maxcosine < cosine_similarity:
        maxcosine = cosine_similarity
        cosfilename = filename

    print("%s filename %f euclediandist %f cosine_similarity"
          % (filename, eucldist, cosine_similarity))
    print("%s minfilename %f mineuclediandist %s cosfilename "
          "%f maxcosinesimilarity" % (minfilename, mindist,
                                      cosfilename, maxcosine))

##############################################################################









