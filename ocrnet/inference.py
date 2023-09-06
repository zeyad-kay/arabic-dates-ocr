import datetime
import os
import tensorflow as tf
from options import load_test_options
from model import build_model
from dataset import ArabicDatesVocabulary, ArabicDatesDataset

if __name__ == "__main__":

    parser = load_test_options()

    args = parser.parse_args()

    batch_size = args.batch_size
    DATA_DIR = args.test_dataset

    filenames = sorted(os.listdir(DATA_DIR))
    images = list()
    text = list()
    for f in filenames:
      if ".jpg" in f:
        images.append(os.path.join(DATA_DIR ,f))
      elif ".txt" in f:
        text.append(os.path.join(DATA_DIR ,f))

    log_dir = "logs/eval/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    test_ds = ArabicDatesDataset(images, text, batch_size)
    
    if args.model:
      print("Loading Pre-trained Model...")
      model = tf.keras.models.load_model(args.model)
    
    else:
      print("Building Model...")
      model = build_model(ArabicDatesVocabulary.size())


    print("Infrencing...")

    model.evaluate(x=test_ds, callbacks=[tensorboard_callback])

    print("Examples")
    
    x,y=test_ds[0]
    
    probs = tf.math.argmax(res, axis=1, output_type=tf.int32)
    
    print(tf.reshape(probs, (res.shape[0]//10,10)))
