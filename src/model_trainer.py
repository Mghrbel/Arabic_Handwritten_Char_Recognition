from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def train(model, BATCH_SIZE = 32, train_dir = '../data/train_labeled/') :
    dataGenerator = ImageDataGenerator(validation_split=0.2)
    
    train_gen = dataGenerator.flow_from_directory(train_dir,
                                                  batch_size=BATCH_SIZE,
                                                  target_size=(32, 32),
                                                  subset='training')
    
    val_gen = dataGenerator.flow_from_directory(train_dir,
                                                batch_size=BATCH_SIZE,
                                                target_size=(32, 32),
                                                subset='validation') 
    
    checkpoint_cb = ModelCheckpoint(filepath='../models/model.h5', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
    early_stopping_cb = EarlyStopping(patience=50, restore_best_weights=True)
    
    model.fit(train_gen,
              epochs = 100,
              steps_per_epoch  = train_gen.samples // BATCH_SIZE,
              validation_data  = val_gen,
              validation_steps = val_gen.samples // BATCH_SIZE,
              callbacks = [checkpoint_cb, early_stopping_cb] )