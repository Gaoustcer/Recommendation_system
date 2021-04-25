from deepctr.feature_column import build_input_features
from deepctr.inputs import combined_dnn_input, create_embedding_matrix
from deepctr.layers.core import PredictionLayer, DNN
from tensorflow.python.keras.models import Model
from deepmatch.inputs import input_from_feature_columns
from deepmatch.layers.core import Similarity
import construct_model
user_feature_column=construct_model.gen_user_column()
