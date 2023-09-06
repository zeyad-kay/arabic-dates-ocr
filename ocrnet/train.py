import datetime
import os
import tensorflow as tf
from options import load_train_options
from model import build_model
from dataset import ArabicDatesVocabulary, ArabicDatesDataset

if __name__ == "__main__":

    parser = load_train_options()

    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    DATA_DIR = args.train_dataset

    filenames = sorted(os.listdir(DATA_DIR))
    images = list()
    text = list()
    for f in filenames:
      if ".jpg" in f:
        images.append(os.path.join(DATA_DIR ,f))
      elif ".txt" in f:
        text.append(os.path.join(DATA_DIR ,f))

    train_ds = ArabicDatesDataset(images, text, batch_size)
    
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    if args.model:
      print("Loading Pre-trained Model...")
      model = tf.keras.models.load_model(args.model)
       
    
    else:
      print("Building Model...")
      model = build_model(ArabicDatesVocabulary.size())


    print("Training...")
    # history = model.fit(x=train_ds, validation_data=val_ds, epochs=1, workers=2, use_multiprocessing=True, verbose=2, callbacks=[tensorboard_callback])
    history = model.fit(x=train_ds, epochs=epochs, callbacks=[tensorboard_callback])
    
    print("Saving Model...")
    
    model.save(args.output_path)
  