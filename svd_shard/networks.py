# from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, ReLU, MaxPooling2D, Flatten



class ActorModel(Sequential):
    def __init__(self, input_space, output_space, depths):
        super().__init__()

        self.model = self.actor_model(input_space, output_space, depths)

    def actor_model(self, input_space, output_space, depths):
        """
        Model used in the paper "IMPALA: Scalable Distributed Deep-RL with 
        Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
        """
        def conv_sequence(model, idx, depth):
            model.add(Conv2D(filters=depth, kernel_size=3, padding='same', name=f"conv_{idx}"))
            model.add(MaxPooling2D(pool_size=3, strides=2, padding='same', name=f"pool_{idx}"))
            return model
            
        model = Sequential()
        model.add(Input(shape=input_space, name="input_layer"))

        for idx, depth in enumerate(depths):
            model = conv_sequence(model, idx, depth)

        model.add(Flatten(name="flatten"))
        model.add(ReLU(name="ReLU"))
        model.add(Dense(output_space, activation="relu", name="dense_output"))
        # model.add(Dense(output_space, activation="softmax", name="dense_output"))

        return model

    def call(self, state):
        scaled_state = tf.cast(state, tf.float32) / 255.
        return self.model(scaled_state)


class CriticModel(Sequential):
    def __init__(self, input_space, depths):
        super().__init__()

        self.model = self.critic_model(input_space, depths)

    def critic_model(self, input_space, depths):
        """
        Model used in the paper "IMPALA: Scalable Distributed Deep-RL with 
        Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
        """
        def conv_sequence(model, idx, depth):
            model.add(Conv2D(filters=depth, kernel_size=3, padding='same', name=f"conv_{idx}"))
            model.add(MaxPooling2D(pool_size=3, strides=2, padding='same', name=f"pool_{idx}"))
            return model
            
        model = Sequential()
        model.add(Input(shape=input_space))

        for idx, depth in enumerate(depths):
            model = conv_sequence(model, idx, depth)

        model.add(Flatten(name="flatten"))
        model.add(ReLU(name="ReLU"))
        model.add(Dense(1, activation="relu", name="dense_output"))

        return model

    def call(self, state):
        scaled_state = tf.cast(state, tf.float32) / 255.
        return self.model(scaled_state)


# def actor_model(input_space, output_space, depths):
#     """
#     Model used in the paper "IMPALA: Scalable Distributed Deep-RL with 
#     Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
#     """
#     def conv_sequence(model, idx, depth):
#         model.add(Conv2D(filters=depth, kernel_size=3, padding='same', name=f"conv_{idx}"))
#         model.add(MaxPooling2D(pool_size=3, strides=2, padding='same', name=f"pool_{idx}"))
#         return model
        
#     model = Sequential()
#     model.add(Input(shape=input_space, name="input_layer"))

#     for idx, depth in enumerate(depths):
#         model = conv_sequence(model, idx, depth)

#     model.add(Flatten(name="flatten"))
#     model.add(ReLU(name="ReLU"))
#     model.add(Dense(output_space, activation="relu", name="dense_output"))

#     return model

# def critic_model(input_space, depths):
#     """
#     Model used in the paper "IMPALA: Scalable Distributed Deep-RL with 
#     Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
#     """
#     def conv_sequence(model, idx, depth):
#         model.add(Conv2D(filters=depth, kernel_size=3, padding='same', name=f"conv_{idx}"))
#         model.add(MaxPooling2D(pool_size=3, strides=2, padding='same', name=f"pool_{idx}"))
#         return model
        
#     model = Sequential()
#     model.add(Input(shape=input_space))

#     for idx, depth in enumerate(depths):
#         model = conv_sequence(model, idx, depth)

#     model.add(Flatten(name="flatten"))
#     model.add(ReLU(name="ReLU"))
#     model.add(Dense(1, activation="relu", name="dense_output"))

#     return model




# class ActorNetwork(keras.Model):
#     def __init__(self, n_actions, fc1_dims=256, fc2_dims=256):
#         super(ActorNetwork, self).__init__()

#         self.fc1 = Dense(fc1_dims, activation='relu')
#         self.fc2 = Dense(fc2_dims, activation='relu')
#         self.fc3 = Dense(n_actions, activation='softmax')

#     def call(self, state):
#         x = self.fc1(state)
#         x = self.fc2(x)
#         x = self.fc3(x)

#         return x

# class ActorModel(Sequential):
#     """
#     Modified model, but original used in the paper "IMPALA: Scalable Distributed Deep-RL with 
#     Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561

#     """
#     def __init__(self, input_space, output_space, depths):
#         super(ActorModel, self).__init__()


#     def conv_sequence(model, idx, depth):
#         model.add(Conv2D(filters=depth, kernel_size=3, padding='same', name=f"conv_{idx}"))
#         model.add(MaxPooling2D(pool_size=3, strides=2, padding='same', name=f"pool_{idx}"))
#         return model
        
#     model.add(Input(shape=input_space))

# #     for idx, depth in enumerate(depths):
# #         model = conv_sequence(model, idx, depth)

# #     model.add(Flatten(name="flatten"))
# #     model.add(ReLU(name="ReLU"))
# #     model.add(Dense(output_space, activation="relu", name="dense_output"))

# #     return model


# class CriticNetwork(keras.Model):
#     def __init__(self, fc1_dims=256, fc2_dims=256):
#         super(CriticNetwork, self).__init__()
#         self.fc1 = Dense(fc1_dims, activation='relu')
#         self.fc2 = Dense(fc2_dims, activation='relu')
#         self.q = Dense(1, activation=None)

#     def call(self, state):
#         x = self.fc1(state)
#         x = self.fc2(x)
#         q = self.q(x)

#         return q
