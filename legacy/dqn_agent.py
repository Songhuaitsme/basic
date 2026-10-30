# 文件功能：实现旧版 DQN、优先经验回放、目标网络同步及模型保存加载。
# 主要内容：PrioritizedReplayBuffer、DQNAgent、_parse_major_minor、_ensure_runtime_compatibility。
# 模块上游（导入关系）：legacy.evaluate、legacy.gnn_agent、legacy.train。
# 模块下游（项目依赖）：shared.config。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

import random
from collections import deque

import numpy as np
import tensorflow as tf
from google.protobuf import __version__ as PROTOBUF_VERSION

from shared import config


# 作用：从版本字符串提取主次版本。
# 上游（直接调用）：legacy.dqn_agent._ensure_runtime_compatibility。
# 下游调用：version.split、char.isdigit、''.join。
def _parse_major_minor(version: str):
    parts = []
    for token in version.split('.'):
        digits = []
        for char in token:
            if char.isdigit():
                digits.append(char)
            else:
                break
        if not digits:
            break
        parts.append(int(''.join(digits)))
        if len(parts) == 2:
            break
    while len(parts) < 2:
        parts.append(0)
    return tuple(parts[:2])


# 作用：检查旧版训练依赖的运行时兼容性。
# 上游（直接调用）：legacy.dqn_agent.DQNAgent.__init__、legacy.gnn_agent.GNNAgent.__init__。
# 下游调用：_parse_major_minor。
def _ensure_runtime_compatibility():
    if _parse_major_minor(tf.__version__) < (2, 11) and _parse_major_minor(PROTOBUF_VERSION)[0] >= 4:
        raise RuntimeError(
            f"Detected incompatible TensorFlow/protobuf versions: tensorflow {tf.__version__}, "
            f"protobuf {PROTOBUF_VERSION}. For TensorFlow < 2.11, please install protobuf==3.20.*."
        )


class PrioritizedReplayBuffer:
    """Proportional prioritized replay with lightweight numpy sampling."""

    # 作用：保存依赖与配置，建立实例初始状态。
    # 上游（直接调用）：legacy.dqn_agent.DQNAgent.__init__。
    # 调用约定：构造实例时由 Python 调用。
    def __init__(self, capacity, alpha=0.6, epsilon=1e-5):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.buffer = []
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.position = 0

    # 作用：返回当前保存的记录数量。
    # 调用约定：由 len() 或容器协议调用。
    def __len__(self):
        return len(self.buffer)

    # 作用：向缓冲区或统计容器追加记录。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    def add(self, experience, priority=None):
        if priority is None:
            current = self.priorities[:len(self.buffer)]
            priority = float(np.max(current)) if len(current) > 0 else 1.0
        priority = max(float(priority), self.epsilon)

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    # 作用：从经验缓冲区随机采样训练批次。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：weights.astype。
    def sample(self, batch_size, beta):
        size = len(self.buffer)
        priorities = self.priorities[:size]
        scaled = np.power(np.maximum(priorities, self.epsilon), self.alpha)
        probs = scaled / np.sum(scaled)

        indices = np.random.choice(size, batch_size, replace=size < batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        weights = np.power(size * probs[indices], -beta)
        weights = weights / np.max(weights)
        return samples, indices, weights.astype(np.float32)

    # 作用：按 TD 误差更新经验采样优先级。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = max(float(priority), self.epsilon)


class DQNAgent:
    # 作用：保存依赖与配置，建立实例初始状态。
    # 上游（直接调用）：legacy.evaluate.create_agent、legacy.train.train。
    # 调用约定：构造实例时由 Python 调用。
    # 下游调用：_ensure_runtime_compatibility、PrioritizedReplayBuffer、self._build_model、self.update_target_model。
    def __init__(self, state_dim=None, action_dim=None):
        _ensure_runtime_compatibility()
        self.state_dim = state_dim if state_dim is not None else config.INPUT_DIM
        self.action_dim = action_dim if action_dim is not None else config.NODE_NUM

        self.use_per = getattr(config, 'PER_ENABLED', True)
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(
                config.MEMORY_CAPACITY,
                alpha=getattr(config, 'PER_ALPHA', 0.6),
                epsilon=getattr(config, 'PER_EPSILON', 1e-5)
            )
        else:
            self.memory = deque(maxlen=config.MEMORY_CAPACITY)

        self.epsilon = config.EPSILON_START
        self.replay_steps = 0

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    # 作用：构建旧版 DQN 网络。
    # 上游（直接调用）：legacy.dqn_agent.DQNAgent.__init__。
    # 下游调用：model.compile。
    def _build_model(self):
        inputs = tf.keras.Input(shape=(self.state_dim,), name='state')
        x = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform')(inputs)
        x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform')(x)

        value = tf.keras.layers.Dense(
            getattr(config, 'DUELING_HIDDEN_DIM', 256),
            activation='relu',
            kernel_initializer='he_uniform'
        )(x)
        value = tf.keras.layers.Dense(1, activation='linear', name='state_value')(value)

        advantage = tf.keras.layers.Dense(
            getattr(config, 'DUELING_HIDDEN_DIM', 256),
            activation='relu',
            kernel_initializer='he_uniform'
        )(x)
        advantage = tf.keras.layers.Dense(self.action_dim, activation='linear', name='action_advantage')(advantage)

        centered_advantage = tf.keras.layers.Lambda(
            lambda a: a - tf.reduce_mean(a, axis=1, keepdims=True),
            name='centered_advantage'
        )(advantage)
        outputs = tf.keras.layers.Add(name='dueling_q')([value, centered_advantage])

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='DuelingDoubleDQN')
        model.compile(
            loss=tf.keras.losses.Huber(),
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=config.LEARNING_RATE,
                clipnorm=1.0
            )
        )
        return model

    # 作用：将在线模型参数同步到目标模型。
    # 上游（直接调用）：legacy.dqn_agent.DQNAgent.__init__、legacy.dqn_agent.DQNAgent.load。
    # 下游调用：self.target_model.set_weights、self.model.get_weights。
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # 作用：根据状态执行探索或按价值选择有效动作。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：self._mask_q_values、self.model(state_tensor, training=False).numpy、self.model。
    def act(self, state, valid_actions=None):
        if np.random.rand() <= self.epsilon:
            if valid_actions is not None and len(valid_actions) > 0:
                return random.choice(valid_actions)
            return random.randint(0, self.action_dim - 1)

        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        q_values = self.model(state_tensor, training=False).numpy()[0]
        q_values = self._mask_q_values(q_values[None, :], [valid_actions])[0]
        return int(np.argmax(q_values))

    # 作用：计算新经验的初始回放优先级。
    # 上游（直接调用）：legacy.dqn_agent.DQNAgent.remember。
    def _initial_priority(self, reward, info):
        priority = abs(float(reward)) + 1.0
        if not isinstance(info, dict):
            return priority

        status = info.get('status')
        if status not in (None, 'Success', 'Deferred'):
            priority += getattr(config, 'PER_FAILURE_PRIORITY_BOOST', 4.0)

        constraint_costs = info.get('constraint_costs', {})
        if constraint_costs:
            priority += getattr(config, 'PER_CONSTRAINT_PRIORITY_BOOST', 3.0) * sum(
                float(v) for v in constraint_costs.values()
            )

        components = info.get('reward_components', {})
        if components:
            cec_proxy = np.mean([
                max(0.0, float(components.get('R_green', 0.0))),
                max(0.0, float(components.get('R_cost', 0.0))),
                max(0.0, float(components.get('R_balance', 0.0))),
                max(0.0, float(components.get('R_success', 0.0))),
            ])
            if cec_proxy < 0.25:
                priority += getattr(config, 'PER_LOW_CECI_PRIORITY_BOOST', 2.0)

        return priority

    # 作用：把状态、动作、奖励及下一状态保存到回放。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：self._initial_priority。
    def remember(self, state, action, reward, next_state, done,
                 valid_actions=None, next_valid_actions=None, info=None):
        experience = (
            np.asarray(state, dtype=np.float32).copy(),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32).copy(),
            bool(done),
            list(valid_actions) if valid_actions is not None else None,
            list(next_valid_actions) if next_valid_actions is not None else None,
        )

        if self.use_per:
            self.memory.add(experience, priority=self._initial_priority(reward, info))
        else:
            self.memory.append(experience)

    # 作用：屏蔽不可行动作的价值。
    # 上游（直接调用）：legacy.dqn_agent.DQNAgent.act、legacy.dqn_agent.DQNAgent.replay。
    @staticmethod
    def _mask_q_values(q_values, valid_actions_batch):
        masked = np.array(q_values, copy=True)
        if valid_actions_batch is None:
            return masked

        for row, valid_actions in enumerate(valid_actions_batch):
            if valid_actions is None:
                continue
            if len(valid_actions) == 0:
                masked[row, :] = -np.inf
                continue
            mask = np.ones(masked.shape[1], dtype=bool)
            mask[np.asarray(valid_actions, dtype=np.int32)] = False
            masked[row, mask] = -np.inf
        return masked

    # 作用：按训练进度计算重要性采样修正系数。
    # 上游（直接调用）：legacy.dqn_agent.DQNAgent.replay。
    def _current_beta(self):
        beta_start = getattr(config, 'PER_BETA_START', 0.4)
        beta_frames = max(1, getattr(config, 'PER_BETA_FRAMES', 120000))
        progress = min(1.0, self.replay_steps / beta_frames)
        return beta_start + progress * (1.0 - beta_start)

    # 作用：采样经验，更新网络及经验优先级。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：self._current_beta、self._mask_q_values、self.memory.sample、self.model(states_tensor, training=False).numpy、self.model 等。
    def replay(self):
        if len(self.memory) < config.BATCH_SIZE:
            return None

        self.replay_steps += 1
        if self.use_per:
            minibatch, sample_indices, sample_weights = self.memory.sample(
                config.BATCH_SIZE,
                beta=self._current_beta()
            )
        else:
            minibatch = random.sample(self.memory, config.BATCH_SIZE)
            sample_indices = None
            sample_weights = np.ones(config.BATCH_SIZE, dtype=np.float32)

        states = np.asarray([item[0] for item in minibatch], dtype=np.float32)
        actions = np.asarray([item[1] for item in minibatch], dtype=np.int32)
        rewards = np.asarray([item[2] for item in minibatch], dtype=np.float32)
        next_states = np.asarray([item[3] for item in minibatch], dtype=np.float32)
        dones = np.asarray([item[4] for item in minibatch], dtype=np.float32)
        next_valid_actions = [item[6] for item in minibatch]

        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states_tensor = tf.convert_to_tensor(next_states, dtype=tf.float32)

        targets = self.model(states_tensor, training=False).numpy()

        next_q_current = self.model(next_states_tensor, training=False).numpy()
        next_q_current = self._mask_q_values(next_q_current, next_valid_actions)
        best_next_actions = np.argmax(next_q_current, axis=1)

        next_q_target = self.target_model(next_states_tensor, training=False).numpy()
        batch_indices = np.arange(config.BATCH_SIZE)
        max_next_q = next_q_target[batch_indices, best_next_actions]
        td_targets = rewards + (1.0 - dones) * config.GAMMA * max_next_q

        td_errors = td_targets - targets[batch_indices, actions]
        targets[batch_indices, actions] = td_targets

        loss = self.model.train_on_batch(states, targets, sample_weight=sample_weights)

        if self.use_per:
            self.memory.update_priorities(
                sample_indices,
                np.abs(td_errors) + getattr(config, 'PER_EPSILON', 1e-5)
            )

        if self.epsilon > config.EPSILON_MIN:
            self.epsilon = max(config.EPSILON_MIN, self.epsilon * config.EPSILON_DECAY)

        return loss[0] if isinstance(loss, list) else float(loss)

    # 作用：保存模型参数及相关状态。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：self.model.save。
    def save(self, filepath):
        self.model.save(filepath)
        print(f"Model saved to: {filepath}")

    # 作用：加载模型参数及相关状态。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：self.update_target_model、self.model.compile、self.target_model.compile。
    def load(self, filepath):
        if os.path.exists(filepath):
            try:
                self.model = tf.keras.models.load_model(filepath, compile=False)
                self.model.compile(
                    loss=tf.keras.losses.Huber(),
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=config.LEARNING_RATE,
                        clipnorm=1.0
                    )
                )
                self.target_model = tf.keras.models.load_model(filepath, compile=False)
                self.target_model.compile(
                    loss=tf.keras.losses.Huber(),
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=config.LEARNING_RATE,
                        clipnorm=1.0
                    )
                )
                self.update_target_model()
                self.epsilon = 0.0
                print(f"Model loaded from: {filepath}")
            except Exception as exc:
                raise RuntimeError(f"Failed to load model: {filepath}") from exc
        else:
            print(f"Error: model file not found {filepath}")
