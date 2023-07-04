import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tensorflow import keras
import Meshpkg.params as p

class NNmodel:

    def __init__(self):
        self.num_neighbor = p.num_neighbor
        self.n_actions = p.n_actions
        self.hidden_node = p.hidden_node
        self.loss_fn = p.loss_fn
        self.optimizer = p.optimizer
        self.state_input = (self.num_neighbor*2+1 -2)*2
        
        if p.act_shape == 0: # [1,2]
            self.action_input = 2
        elif p.act_shape == 1: # [1, 625]
            self.action_input = 25 ** 2
        elif p.act_shape == 2: # [1,20]
            self.action_input = 5 * 2 * 2
    """
    Dense
    """
    def dense(self): #model weight 초기화
        model = keras.models.Sequential([
            keras.layers.Input(shape =(self.state_input, ) ),
            keras.layers.Dense(self.hidden_node),
            keras.layers.Activation(activation = "tanh"),
            keras.layers.Dense(self.hidden_node),
            keras.layers.Activation(activation = "tanh"),
            keras.layers.Dense(self.hidden_node),
            keras.layers.Activation(activation = "tanh"),
            keras.layers.Dense(self.n_actions, 
                               kernel_initializer = keras.initializers.RandomUniform(minval=-1e-3, maxval=1e-3),
                               activation = 'linear'),
        ])

        model.compile(loss = self.loss_fn, optimizer = self.optimizer) # model compiling

        return model
    
    def dense_multi(self): #model weight 초기화

        state = keras.Input(shape = (self.state_input,) )
        
        action = keras.Input(shape = (self.action_input,) )

        layer0 = keras.layers.Concatenate()([state, action])
        layer1 = keras.layers.Dense(self.hidden_node, activation = 'gelu')(layer0)
        layer2 = keras.layers.Dense(self.hidden_node, activation = 'gelu')(layer1)
        layer3 = keras.layers.Dense(self.hidden_node, activation = 'gelu')(layer2)

        Q_values = keras.layers.Dense(self.n_actions, 
                                      kernel_initializer = keras.initializers.RandomUniform(minval=-1e-3, maxval=1e-3),
                                      activation = 'linear')(layer3)
        
        model = keras.Model(inputs = [state, action], outputs = [Q_values])
        model.compile(loss = self.loss_fn, optimizer = self.optimizer) # model compiling

        return model
    """
    RNN
    """
    def rnn(self):
        model = keras.models.Sequential()
        model.add(keras.layers.LSTM(self.hidden_node, return_sequences=False, input_shape = (self.state_input, 1)))   # return_sequences parameter has to be set True to stack
        model.add(keras.layers.Dense(self.n_actions, kernel_initializer = keras.initializers.RandomUniform(minval=-1e-3, maxval=1e-3)))
        model.compile(loss = self.loss_fn, optimizer = self.optimizer)

        return model
    """
    Dueling 신경망: Q value = State value + Advantage 
    """
    def dueling(self):
        input_states = keras.layers.Input(shape =(self.state_input, ))
        hidden1 = keras.layers.Dense(self.hidden_node, activation = "gelu")(input_states)
        hidden2 = keras.layers.Dense(self.hidden_node, activation = "gelu")(hidden1)
        hidden3 = keras.layers.Dense(self.hidden_node/2, activation = "gelu")(hidden2)

        state_values = keras.layers.Dense(1)(hidden3)
        raw_advantages = keras.layers.Dense(self.n_actions, kernel_initializer = keras.initializers.RandomUniform(minval=-1e-3, maxval=1e-3))(hidden3)

        advantages = raw_advantages - keras.backend.max(raw_advantages)
        Q_values = state_values + advantages

        model = keras.Model(input_states, Q_values)
        model.compile(loss = self.loss_fn, optimizer = self.optimizer)

        return model