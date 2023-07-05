from Meshpkg.Initialize.action_definition import get_action_type
from Meshpkg.Initialize.surface_mesh import get_surf_mesh

from tensorflow import keras

"""
[Mesh shape]
"""
# spline_1 : 하트(점 34개), spline_1_1 : 하트(점 16개)
mesh_name = 'spline_1_1'

surf_mesh = get_surf_mesh(mesh_name)
surf_length = len(surf_mesh)
first_layer = 0.005
growth_rate = 1.1
num_layer = 30

"""
[State, Action input] 
<num_neighbor>
State - 좌/우 점 몇개 들어가는지 
<act_shape>
Neighborhood actions - 어떤 형태로 들어가는지
0: [1,2]      좌/우 action의 action index를 [0,1]범위로 normalize 한 것
1: [1,625]    좌/우 action의 action index를 one-hot encoding한 후 외적한 것
2: [1, 20]    좌/우 action의 action index를 길이: 5차원 one-hot / 각도: 5차원 one-hot 한후 옆으로 붙인 것 
<num_iter>
Neighborhood actions 계산하기 위해 iteration 몇번 돌리는지
"""
num_neighbor = 3
act_shape = 2
num_iter =5

"""
[Action 정의]
"""
action_space = get_action_type(2)
n_actions = len(action_space)

"""
[DQN 관련]
Target Q계산시 할인율
"""
discount_rate = 0.999
# 몇 episode마다 target model update할 건지
episode_target = 5
# target model update 시 (tau)만큼 online model에서, (1 - tau)만큼 target model에서
tau = 0.005
"""
[Replay memory 관련]
Buffer/Batch size
"""
buffer_size = 5000
batch_size = 128

"""
[Policy 관련]
Policy 선택: Epsilon greedy = 0, Softmax = 1
"""
policy = 0
"Softmax policy temperature"
temp = 1
"epsilon greedy policy parameters: 0.01 도달까지  0.999일경우 (4605 episode), 0.99일 경우 (459 episode)"
epsilon_start = 0.99
epsilon_min = 0.01

"""
[Neural network model training 관련]
"""
loss_fn = keras.losses.mean_squared_error
learning_rate = 0.0005
optimizer = keras.optimizers.Adam(learning_rate = learning_rate)
hidden_node = 128

"""
[Plot관련] 
colormap 정의
"""
colormap = ['black','blue','red','purple','turquoise']