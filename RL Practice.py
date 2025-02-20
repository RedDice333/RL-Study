import sys
import random
import numpy as np
from collections import deque

import gym
import torch
import torch.nn as nn
import torch.optim as optim

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QTextEdit
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt

# ---------------------------
# 디바이스 설정 (GPU 사용)
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------
# DQN 모델 및 유틸리티 함수
# ---------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # 상태와 next_state는 np.stack()을 사용해 고정 shape 배열로 만듦
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.stack(states), np.array(actions), np.array(rewards), np.stack(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)

def select_action(state, policy_net, epsilon, action_dim):
    if random.random() < epsilon:
        return random.randrange(action_dim)
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
        return q_values.argmax().item()

def compute_td_loss(batch, policy_net, target_net, optimizer, GAMMA):
    states, actions, rewards, next_states, dones = batch

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).to(device)

    q_values = policy_net(states)
    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = target_net(next_states)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = rewards + GAMMA * next_q_value * (1 - dones)

    loss = nn.MSELoss()(q_value, expected_q_value)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# ---------------------------
# QThread를 활용한 학습 쓰레드
# ---------------------------
class TrainingThread(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, env_name, parent=None):
        super(TrainingThread, self).__init__(parent)
        self.env_name = env_name
        self.policy_net = None  # 학습 완료 후 저장

    def run(self):
        # 하이퍼파라미터 설정
        EPISODES = 500
        LEARNING_RATE = 0.001
        GAMMA = 0.99
        BATCH_SIZE = 64
        MEMORY_SIZE = 10000
        TARGET_UPDATE = 10
        EPS_START = 1.0
        EPS_END = 0.01
        EPS_DECAY = 0.995

        # 학습용 환경 생성 (렌더링은 필요없음)
        env = gym.make(self.env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # 모델 생성 후 GPU로 이동
        policy_net = DQN(state_dim, action_dim).to(device)
        target_net = DQN(state_dim, action_dim).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
        replay_buffer = ReplayBuffer(MEMORY_SIZE)
        epsilon = EPS_START

        for episode in range(EPISODES):
            # 최신 Gym API: reset()는 (observation, info)를 반환함
            state, _ = env.reset()
            total_reward = 0
            done = False
            while not done:
                action = select_action(state, policy_net, epsilon, action_dim)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                if done:
                    reward = -1.0  # 종료 시 보상 조정
                replay_buffer.push(state, action, reward, next_state, done)
                state = next_state

                if len(replay_buffer) > BATCH_SIZE:
                    batch = replay_buffer.sample(BATCH_SIZE)
                    compute_td_loss(batch, policy_net, target_net, optimizer, GAMMA)

            epsilon = max(EPS_END, epsilon * EPS_DECAY)
            if episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
            self.progress_signal.emit(f"Episode {episode+1:03d} | Total Reward: {total_reward} | ε: {epsilon:.3f}")
        env.close()

        # 학습 완료 후 학습된 모델 저장
        self.policy_net = policy_net
        self.finished_signal.emit()

# ---------------------------
# 데모 창 (CartPole 실행 화면)
# ---------------------------
class CartPoleDemo(QWidget):
    def __init__(self, policy_net, env_name):
        super(CartPoleDemo, self).__init__()
        self.policy_net = policy_net
        self.env_name = env_name
        # 데모용 환경 생성 (rgb_array 모드)
        self.env = gym.make(env_name, render_mode='rgb_array')
        self.state, _ = self.env.reset()
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.step_simulation)

    def init_ui(self):
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(600, 400)
        self.start_button = QPushButton("Start Demo", self)
        self.stop_button = QPushButton("Stop Demo", self)
        self.start_button.clicked.connect(self.start_demo)
        self.stop_button.clicked.connect(self.stop_demo)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.start_button)
        btn_layout.addWidget(self.stop_button)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self.setWindowTitle("CartPole Demo")

    def start_demo(self):
        self.state, _ = self.env.reset()
        self.timer.start(50)  # 50ms 간격 업데이트

    def stop_demo(self):
        self.timer.stop()

    def step_simulation(self):
        # 학습된 정책대로 행동 선택 (탐험 없이, ε=0)
        action = select_action(self.state, self.policy_net, epsilon=0.0, action_dim=self.env.action_space.n)
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        self.state = next_state

        # rgb_array 모드로 받은 프레임을 QImage로 변환
        frame = self.env.render()
        # numpy 배열을 연속적인 배열로 변환하여 QImage 생성 문제 해결
        frame = np.ascontiguousarray(frame)
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

        if done:
            self.timer.stop()
            self.state, _ = self.env.reset()

# ---------------------------
# 메인 윈도우 (학습 및 데모 제어)
# ---------------------------
class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.env_name = "CartPole-v1"
        self.policy_net = None
        self.training_thread = None
        self.init_ui()

    def init_ui(self):
        self.train_button = QPushButton("Train Model", self)
        self.train_button.clicked.connect(self.start_training)

        self.start_demo_button = QPushButton("Start Demo", self)
        self.start_demo_button.clicked.connect(self.start_demo)
        self.start_demo_button.setEnabled(False)

        self.log_text = QTextEdit(self)
        self.log_text.setReadOnly(True)

        layout = QVBoxLayout()
        layout.addWidget(self.train_button)
        layout.addWidget(self.start_demo_button)
        layout.addWidget(self.log_text)
        self.setLayout(layout)
        self.setWindowTitle("CartPole DQN - PyQt5")
        self.resize(800, 600)

    def start_training(self):
        self.log_text.append("Training started...")
        self.training_thread = TrainingThread(self.env_name)
        self.training_thread.progress_signal.connect(self.update_log)
        self.training_thread.finished_signal.connect(self.training_finished)
        self.training_thread.start()

    def update_log(self, message):
        self.log_text.append(message)

    def training_finished(self):
        self.log_text.append("Training finished!")
        self.policy_net = self.training_thread.policy_net
        self.start_demo_button.setEnabled(True)

    def start_demo(self):
        if self.policy_net is None:
            self.log_text.append("Model is not trained yet!")
            return
        self.demo_window = CartPoleDemo(self.policy_net, self.env_name)
        self.demo_window.show()

# ---------------------------
# 메인 실행부
# ---------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
