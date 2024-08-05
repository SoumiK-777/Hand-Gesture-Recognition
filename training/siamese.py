import argparse
import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.utils import Sequence
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder, target_size=(128, 128)):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = load_img(img_path, target_size=target_size)
            img = img_to_array(img)
            images.append(img)
    return np.array(images)

def create_image_paths_and_labels(dataset_folder):
    classes = os.listdir(dataset_folder)
    image_paths = []
    labels = []
    
    for class_name in classes:
        class_path = os.path.join(dataset_folder, class_name)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                if os.path.isfile(img_path):
                    image_paths.append(img_path)
                    labels.append(class_name)
    
    return image_paths, labels

class PairGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size=32, target_size=(128, 128)):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.class_indices = {label: np.where(np.array(labels) == label)[0] for label in set(labels)}
        self.classes = list(self.class_indices.keys())

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        
        pair_images1 = []
        pair_images2 = []
        labels = []

        for i, img_path in enumerate(batch_paths):
            img = load_img(img_path, target_size=self.target_size)
            img = img_to_array(img) / 255.0
            label = batch_labels[i]
            
            # Create positive pair
            same_class_indices = self.class_indices[label]
            if len(same_class_indices) > 1:
                pair_idx = np.random.choice(same_class_indices)
                while pair_idx == i:
                    pair_idx = np.random.choice(same_class_indices)
                img2 = load_img(self.image_paths[pair_idx], target_size=self.target_size)
                img2 = img_to_array(img2) / 255.0
                
                pair_images1.append(img)
                pair_images2.append(img2)
                labels.append(1)
            
            # Create negative pair
            different_class = np.random.choice([cls for cls in self.classes if cls != label])
            different_class_idx = np.random.choice(self.class_indices[different_class])
            img2 = load_img(self.image_paths[different_class_idx], target_size=self.target_size)
            img2 = img_to_array(img2) / 255.0

            pair_images1.append(img)
            pair_images2.append(img2)
            labels.append(0)

        return [np.array(pair_images1), np.array(pair_images2)], np.array(labels).astype('float32')

    def on_epoch_end(self):
        np.random.shuffle(self.image_paths)

def build_base_network(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(32, (10, 10), activation='relu')(input)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (7, 7), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (4, 4), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (4, 4), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(1024, activation='sigmoid')(x)
    return Model(input, x)

def euclidean_distance(vectors):
    x, y = vectors
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    image_paths, labels = create_image_paths_and_labels(args.dataset_folder)
    
    input_shape = (128, 128, 3)
    base_network = build_base_network(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)

    model.compile(loss=contrastive_loss, optimizer=Adam(learning_rate=0.0006), metrics=['accuracy'])

    checkpoint_path = 'checkpoints'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    checkpoint = ModelCheckpoint(os.path.join(checkpoint_path, 'model.h5'), monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)
    
    train_generator = PairGenerator(train_paths, train_labels, batch_size=BATCH_SIZE)
    val_generator = PairGenerator(val_paths, val_labels, batch_size=BATCH_SIZE)

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=[checkpoint]
    )

    model.save('siamese_model.h5')

if __name__ == '__main__':
    main()