import tensorflow as tf
import cv2 as cv
import math
from utils import partition

class ArabicDatesVocabulary:
  """Static class for tokenizing and detokenizing a date sequence to and from labels.
  The class creates a `tf.lookup.StaticHashTable` that maps characters to labels and vice versa.
  """

  vocab = tf.constant(["٠","١","٢","٣","٤","٥","٦","٧","٨","٩","/"])
  # vocab = tf.constant(["0","1","2","3","4","5","6","7","8","9","/"])
  
  vocab_to_labels = tf.lookup.StaticHashTable(
     tf.lookup.KeyValueTensorInitializer(vocab, tf.range(0, vocab.shape[0])),
     default_value=11)
  
  labels_to_vocab = tf.lookup.StaticHashTable(
     tf.lookup.KeyValueTensorInitializer(tf.range(0, vocab.shape[0]), vocab),
     default_value="")

  # Prevent instantiation
  def __new__(cls):
      raise TypeError("This is a static class and cannot be instantiated.")
  
  @staticmethod
  def size():
      return ArabicDatesVocabulary.vocab_to_labels.size().numpy()
  @staticmethod
  def tokenize(text, encoding: str = 'UTF-8') -> tf.Tensor:
    """Converts characters in a text to unique labels
     Args:
         text: sequence of text
     Returns:
         tf.Tensor: Converted labels
     """
    sequence = tf.strings.unicode_split(text, encoding)
    return ArabicDatesVocabulary.vocab_to_labels[sequence]
  @staticmethod
  def detokenize(labels: tf.Tensor) -> tf.Tensor:
    """Converts labels to sequence of characters
     Args:
         labels (tf.Tensor): sequence of characters
     Returns:
         tf.Tensor: Converted sequence
     """
    return ArabicDatesVocabulary.labels_to_vocab[labels]

class ArabicDatesDataset(tf.keras.utils.Sequence):
  """Wrapper class for loading the dataset. `tf.keras.utils.Sequence` is used instead of `tf.data.Dataset`
  because the preprocessing step includes operations that are run on the CPU not the GPU, e.g. contours detection.
  To prevent bottlenecks, `model.fit(..., use_multiprocessing=True)` is used to parallelize the preprocessing.
  """
  def __init__(self, x_set, y_set, batch_size):

    self.batch_size = batch_size
    self.x, self.y = x_set, y_set

  def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)
  
  def __getitem__(self, idx):
    low = idx * self.batch_size
    # Cap upper bound at array length; the last batch may be smaller
    # if the total number of items is not a multiple of batch size.
    high = min(low + self.batch_size, len(self.x))
    batch_x = self.x[low:high]
    batch_y = self.y[low:high]

    # partition a single image to 10 64x64 images corresponding
    # to each character
    imgs = tf.concat([partition(cv.imread(x,0)) for x in batch_x], axis=0)
    
    # tokenize the date into labels
    labels = ArabicDatesVocabulary.tokenize([tf.io.read_file(y) for y in batch_y])
    
    return imgs, labels.numpy().flatten()