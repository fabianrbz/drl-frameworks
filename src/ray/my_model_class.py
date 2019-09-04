from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
import tensorflow as tf
from ray.rllib.agents.dqn.distributional_q_model import DistributionalQModel
from gym import spaces
import numpy as np
import pdb
from ray.rllib.models.tf.misc import normc_initializer

slim = tf.contrib.slim

class MyModelClass(DistributionalQModel, TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kw):
      super(MyModelClass, self).__init__(obs_space, action_space, num_outputs, model_config, name, **kw)

      # inputs   = tf.keras.layers.Input(shape=(12, 16), name="input")
      # layer_1  = tf.keras.layers.Conv2D(32, kernel_size=4, name="conv1")(inputs)
      # layer_2  = tf.keras.layers.Conv2D(64, kernel_size=4, name="conv2")(layer_1)
      # layer_3  = tf.keras.layers.Conv2D(64, kernel_size=4, name="conv3")(layer_2)
      # flat     = tf.keras.layers.Flatten()(layer_3)
      # output   = tf.keras.layers.Dense(512)(inputs)
      # q_values = tf.keras.layers.Dense(num_outputs)(output)

      # self.base_model = tf.keras.Model(inputs, [output, q_values])

      self.base_model = FullyConnectedNetwork(
              spaces.Box(low=-np.finfo(np.float32).max, high=np.finfo(np.float32).max, shape=(192,)),
              action_space, 16,
              model_config, name)
      self.register_variables(self.base_model.variables())
      # self.inputs   = tf.keras.layers.Input(shape=(192,), name="input")
      # output   = tf.keras.layers.Dense(16, activation=tf.nn.relu, kernel_initializer=normc_initializer(1.0))(self.inputs)
      # self.base_model = tf.keras.Model(self.inputs, [output])
      # print(self.base_model.summary())
      # self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        avail_actions = input_dict["obs"]["avail_actions"]

        # Compute the predicted action embedding
        action_embed, _ = self.base_model({
            "obs": input_dict["obs"]["indexer"]
        })

        # Expand the model output to [BATCH, 1, EMBED_SIZE]. Note that the
        # avail actions tensor is of shape [BATCH, MAX_ACTIONS, EMBED_SIZE].
        #    shape=(?, 1, 2)
        # intent_vector = tf.expand_dims(action_embed, 16)

        # Batch dot product => shape of logits is [BATCH, MAX_ACTIONS].
        # action_logits = tf.reduce_sum(avail_actions * action_embed, axis=2)
        output = tf.math.multiply(avail_actions, action_embed)

        # Mask out invalid actions (use tf.float32.min for stability)
        # inf_mask = tf.maximum(tf.log(action_mask), tf.float32.min)

        return output, state

    def value_function(self):
      return self.base_model.value_function()

