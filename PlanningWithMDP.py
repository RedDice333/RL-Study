import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QDoubleSpinBox, QSpinBox, QLabel, QDialog, QDialogButtonBox, QFormLayout, QCheckBox
)
from PyQt5.QtGui import QPainter, QPen, QBrush, QFont, QPolygonF
from PyQt5.QtCore import Qt, QPointF, pyqtSignal

# -------------------------------------------------
# CellEditDialog: 셀 값을 편집하는 다이얼로그
# -------------------------------------------------
class CellEditDialog(QDialog):
    def __init__(self, current_value, current_locked, parent=None):
        super(CellEditDialog, self).__init__(parent)
        self.setWindowTitle("셀 편집")
        self.new_value = current_value
        self.locked = current_locked

        # 폼 레이아웃: 값(QDoubleSpinBox)와 고정 여부(QCheckBox)
        layout = QFormLayout(self)
        self.value_spin = QDoubleSpinBox(self)
        self.value_spin.setRange(-1000, 1000)
        self.value_spin.setDecimals(2)
        self.value_spin.setValue(current_value)
        layout.addRow("값:", self.value_spin)

        self.lock_checkbox = QCheckBox("고정 (업데이트 시 변경 안됨)", self)
        self.lock_checkbox.setChecked(current_locked)
        layout.addRow("고정:", self.lock_checkbox)

        # OK / Cancel 버튼
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def accept(self):
        self.new_value = self.value_spin.value()
        self.locked = self.lock_checkbox.isChecked()
        super(CellEditDialog, self).accept()


# -------------------------------------------------
# GridWidget: 격자와 현재 Value, Policy(화살표)를 그리는 위젯
#   - 셀 클릭 시 cellClicked 신호를 발생 (행, 열)
# -------------------------------------------------
class GridWidget(QWidget):
    cellClicked = pyqtSignal(int, int)  # row, col

    def __init__(self, grid_size, parent=None):
        super(GridWidget, self).__init__(parent)
        self.grid_size = grid_size  # (rows, cols)
        self.V = np.zeros(grid_size)
        self.fixed = np.zeros(grid_size, dtype=bool)  # 고정 여부
        self.policy = None          # 정책: {(row, col): action_string}
        self.terminal_state = (grid_size[0]-1, grid_size[1]-1)  # 오른쪽 맨 아래

    def setData(self, V, fixed):
        self.V = V.copy()
        self.fixed = fixed.copy()
        self.update()

    def setPolicy(self, policy):
        self.policy = policy
        self.update()

    def clearPolicy(self):
        self.policy = None
        self.update()

    def paintEvent(self, event):
        qp = QPainter()
        if not qp.begin(self):
            return
        try:
            self.drawGrid(qp)
        finally:
            qp.end()

    def drawGrid(self, qp):
        rows, cols = self.grid_size
        w = int(self.width())
        h = int(self.height())
        if cols == 0 or rows == 0:
            return
        cell_w = w / cols
        cell_h = h / rows

        # 각 셀 배경: 고정된 셀은 연한 회색으로 표시
        for i in range(rows):
            for j in range(cols):
                x = j * cell_w
                y = i * cell_h
                if self.fixed[i, j]:
                    qp.fillRect(int(x), int(y), int(cell_w), int(cell_h), QBrush(Qt.lightGray))

        # 그리드 선 그리기 (좌표를 int로 캐스팅)
        pen = QPen(Qt.black, 2)
        qp.setPen(pen)
        for i in range(rows+1):
            y = i * cell_h
            qp.drawLine(0, int(y), w, int(y))
        for j in range(cols+1):
            x = j * cell_w
            qp.drawLine(int(x), 0, int(x), h)

        # 텍스트 폰트 설정
        font = QFont("Arial", 12)
        qp.setFont(font)

        # 각 셀에 대해 value와 (정책 있으면) 화살표 표시
        for i in range(rows):
            for j in range(cols):
                x = j * cell_w
                y = i * cell_h
                value = self.V[i, j]
                text = f"{value:.2f}"
                qp.drawText(int(x), int(y), int(cell_w), int(cell_h), Qt.AlignCenter, text)

                # 터미널 상태 강조: 녹색 테두리
                if (i, j) == self.terminal_state:
                    pen_temp = QPen(Qt.green, 3)
                    qp.setPen(pen_temp)
                    qp.drawRect(int(x), int(y), int(cell_w), int(cell_h))
                    qp.setPen(QPen(Qt.black, 2))

                # 정책(화살표) 표시 (터미널은 표시하지 않음)
                if self.policy is not None and (i, j) in self.policy and self.policy[(i, j)] is not None:
                    action = self.policy[(i, j)]
                    arrow_len = min(cell_w, cell_h) / 4
                    dx, dy = 0, 0
                    if action == 'up':
                        dy = -arrow_len
                    elif action == 'down':
                        dy = arrow_len
                    elif action == 'left':
                        dx = -arrow_len
                    elif action == 'right':
                        dx = arrow_len
                    center = QPointF(x + cell_w/2, y + cell_h/2)
                    end = QPointF(center.x() + dx, center.y() + dy)
                    qp.setPen(QPen(Qt.red, 2))
                    qp.drawLine(center, end)
                    # 화살표 머리 그리기 (dx, dy가 0이 아니면)
                    if dx != 0 or dy != 0:
                        angle = np.arctan2(dy, dx)
                        size = 5
                        p1 = QPointF(end.x() - size * np.cos(angle - np.pi/6),
                                     end.y() - size * np.sin(angle - np.pi/6))
                        p2 = QPointF(end.x() - size * np.cos(angle + np.pi/6),
                                     end.y() - size * np.sin(angle + np.pi/6))
                        arrow_head = QPolygonF([end, p1, p2])
                        qp.setBrush(QBrush(Qt.red))
                        qp.drawPolygon(arrow_head)
                        qp.setBrush(Qt.NoBrush)
                    qp.setPen(QPen(Qt.black, 2))

    def mousePressEvent(self, event):
        # 클릭한 위치에 해당하는 셀의 (row, col)을 계산 후 신호 발생
        pos = event.pos()
        rows, cols = self.grid_size
        cell_w = self.width() / cols
        cell_h = self.height() / rows
        col = int(pos.x() // cell_w)
        row = int(pos.y() // cell_h)
        # 범위 내라면 신호 방출
        if 0 <= row < rows and 0 <= col < cols:
            self.cellClicked.emit(row, col)


# -------------------------------------------------
# MainWindow: 인터페이스와 업데이트 로직
# -------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("MDP Grid Planner")
        # 기본 격자 크기 (초기: 5x5)
        self.grid_size = (5, 5)
        self.terminal_state = (self.grid_size[0]-1, self.grid_size[1]-1)  # 오른쪽 맨 아래
        self.start_state = (0, 0)  # 왼쪽 맨 위
        self.step_reward = -1
        self.gamma = 0.9  # 기본 감쇠율
        self.theta = 1e-4

        # 초기 상태 가치(V)와 고정 여부(fixed) 배열 설정
        self.V = np.zeros(self.grid_size)
        self.fixed = np.zeros(self.grid_size, dtype=bool)
        self.rewards = self.step_reward * np.ones(self.grid_size)
        self.rewards[self.terminal_state] = 0

        # 액션: 상, 하, 좌, 우
        self.actions = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }

        # 메인 위젯 및 레이아웃 구성
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.main_layout = QVBoxLayout()  # 격자 위젯을 동적으로 교체하기 위해 멤버 변수로 저장
        main_widget.setLayout(self.main_layout)

        # [격자 크기 입력 위젯]
        grid_size_layout = QHBoxLayout()
        grid_size_label = QLabel("격자 크기 (행 x 열):")
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(2, 20)
        self.rows_spin.setValue(self.grid_size[0])
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(2, 20)
        self.cols_spin.setValue(self.grid_size[1])
        grid_size_layout.addWidget(grid_size_label)
        grid_size_layout.addWidget(self.rows_spin)
        grid_size_layout.addWidget(self.cols_spin)
        self.main_layout.addLayout(grid_size_layout)

        # [감쇠율 조절 위젯]
        gamma_layout = QHBoxLayout()
        gamma_label = QLabel("감쇠율 (gamma):")
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0, 1)
        self.gamma_spin.setSingleStep(0.01)
        self.gamma_spin.setDecimals(2)
        self.gamma_spin.setValue(self.gamma)
        self.gamma_spin.valueChanged.connect(self.change_gamma)
        gamma_layout.addWidget(gamma_label)
        gamma_layout.addWidget(self.gamma_spin)
        self.main_layout.addLayout(gamma_layout)

        # 격자 표시 위젯 (초기 격자)
        self.gridWidget = GridWidget(self.grid_size)
        self.gridWidget.setData(self.V, self.fixed)
        self.gridWidget.cellClicked.connect(self.editCell)
        self.main_layout.addWidget(self.gridWidget)

        # [버튼들을 담을 레이아웃]
        button_layout = QHBoxLayout()
        self.main_layout.addLayout(button_layout)

        # [초기화] → 격자 크기와 감쇠율에 따라 다시 초기화 (새 격자 생성)
        btn_reset = QPushButton("초기화")
        btn_reset.clicked.connect(self.reset)
        button_layout.addWidget(btn_reset)

        # [최적 선택으로 다음 Step 계산]
        btn_opt_step = QPushButton("최적 선택으로 다음 Step 계산")
        btn_opt_step.clicked.connect(self.optimal_step)
        button_layout.addWidget(btn_opt_step)

        # [최적 선택으로 수렴까지 계산]
        btn_opt_conv = QPushButton("최적 선택으로 수렴까지 계산")
        btn_opt_conv.clicked.connect(self.optimal_convergence)
        button_layout.addWidget(btn_opt_conv)

        # [확률로 다음 Step 계산]
        btn_prob_step = QPushButton("확률로 다음 Step 계산")
        btn_prob_step.clicked.connect(self.prob_step)
        button_layout.addWidget(btn_prob_step)

        # [확률로 수렴까지 계산]
        btn_prob_conv = QPushButton("확률로 수렴까지 계산")
        btn_prob_conv.clicked.connect(self.prob_convergence)
        button_layout.addWidget(btn_prob_conv)

        # [현 Value에 따른 Policy 표시]
        btn_show_policy = QPushButton("현 Value에 따른 Policy 표시")
        btn_show_policy.clicked.connect(self.show_policy)
        button_layout.addWidget(btn_show_policy)

    # 감쇠율 변경 슬롯
    def change_gamma(self, value):
        self.gamma = value
        print(f"감쇠율 변경: gamma = {self.gamma}")

    # -------------------------------------------------
    # 헬퍼 함수: 격자 환경 내에서 상태 전이 계산
    def step_function(self, state, action):
        new_state = (state[0] + action[0], state[1] + action[1])
        if new_state[0] < 0 or new_state[0] >= self.grid_size[0] or new_state[1] < 0 or new_state[1] >= self.grid_size[1]:
            return state
        return new_state

    # -------------------------------------------------
    # 최적 선택 업데이트: 고정되지 않은 셀에 대해
    # V(s) = max_a { R(s') + gamma * V(s') }
    def optimal_update(self):
        new_V = self.V.copy()
        delta = 0.0
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                state = (i, j)
                # 터미널 또는 고정된 셀은 업데이트하지 않음
                if state == self.terminal_state or self.fixed[i, j]:
                    continue
                values = []
                for action in self.actions.values():
                    next_state = self.step_function(state, action)
                    values.append(self.rewards[next_state] + self.gamma * self.V[next_state])
                best_val = max(values)
                delta = max(delta, abs(best_val - self.V[state]))
                new_V[state] = best_val
        self.V = new_V
        return delta

    # 확률 업데이트: 고정되지 않은 셀에 대해
    # V(s) = (1/|A|) * sum_a { R(s') + gamma * V(s') }
    def probability_update(self):
        new_V = self.V.copy()
        delta = 0.0
        num_actions = len(self.actions)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                state = (i, j)
                if state == self.terminal_state or self.fixed[i, j]:
                    continue
                total = 0.0
                for action in self.actions.values():
                    next_state = self.step_function(state, action)
                    total += self.rewards[next_state] + self.gamma * self.V[next_state]
                avg_val = total / num_actions
                delta = max(delta, abs(avg_val - self.V[state]))
                new_V[state] = avg_val
        self.V = new_V
        return delta

    # 현재 V에 따른 최적 정책 계산: 고정되지 않은 셀만 업데이트
    def compute_policy(self):
        policy = {}
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                state = (i, j)
                if state == self.terminal_state:
                    policy[state] = None
                elif self.fixed[i, j]:
                    policy[state] = None  # 고정된 셀은 정책 변경 없음
                else:
                    best_action = None
                    best_val = -np.inf
                    for action_name, action in self.actions.items():
                        next_state = self.step_function(state, action)
                        val = self.rewards[next_state] + self.gamma * self.V[next_state]
                        if val > best_val:
                            best_val = val
                            best_action = action_name
                    policy[state] = best_action
        return policy

    # -------------------------------------------------
    # 셀 편집 슬롯: GridWidget의 cellClicked 신호 처리
    def editCell(self, row, col):
        # 현재 값과 고정 상태 읽기
        current_value = self.V[row, col]
        current_locked = self.fixed[row, col]
        dialog = CellEditDialog(current_value, current_locked, self)
        if dialog.exec_() == QDialog.Accepted:
            self.V[row, col] = dialog.new_value
            self.fixed[row, col] = dialog.locked
            self.gridWidget.setData(self.V, self.fixed)
            print(f"셀 ({row}, {col}) 수정: 값={dialog.new_value}, 고정={dialog.locked}")

    # -------------------------------------------------
    # 버튼 클릭 슬롯들
    def reset(self):
        # 격자 크기 입력 위젯에서 새 값을 읽음
        new_rows = self.rows_spin.value()
        new_cols = self.cols_spin.value()
        new_grid_size = (new_rows, new_cols)
        if new_grid_size != self.grid_size:
            # 격자 크기가 변경되었으면 새로 초기화
            self.grid_size = new_grid_size
            self.terminal_state = (self.grid_size[0]-1, self.grid_size[1]-1)
            self.V = np.zeros(self.grid_size)
            self.fixed = np.zeros(self.grid_size, dtype=bool)
            self.rewards = self.step_reward * np.ones(self.grid_size)
            self.rewards[self.terminal_state] = 0
            # 기존 gridWidget 제거 후 새로 생성
            self.gridWidget.setParent(None)
            self.gridWidget.deleteLater()
            self.gridWidget = GridWidget(self.grid_size)
            self.gridWidget.setData(self.V, self.fixed)
            self.gridWidget.cellClicked.connect(self.editCell)
            self.main_layout.insertWidget(2, self.gridWidget)
        else:
            # 격자 크기가 동일하면 V와 고정 상태만 초기화
            self.V = np.zeros(self.grid_size)
            self.fixed = np.zeros(self.grid_size, dtype=bool)
            self.rewards = self.step_reward * np.ones(self.grid_size)
            self.rewards[self.terminal_state] = 0
        self.gridWidget.setData(self.V, self.fixed)
        self.gridWidget.clearPolicy()
        print("초기화 완료.")

    def optimal_step(self):
        delta = self.optimal_update()
        self.gridWidget.setData(self.V, self.fixed)
        self.gridWidget.clearPolicy()
        print(f"최적 선택 Step 업데이트: delta = {delta}")

    def optimal_convergence(self):
        iter_count = 0
        while True:
            delta = self.optimal_update()
            iter_count += 1
            if delta < self.theta:
                break
        self.gridWidget.setData(self.V, self.fixed)
        self.gridWidget.clearPolicy()
        print(f"최적 선택 수렴 완료: {iter_count}회, delta = {delta}")

    def prob_step(self):
        delta = self.probability_update()
        self.gridWidget.setData(self.V, self.fixed)
        self.gridWidget.clearPolicy()
        print(f"확률 기반 Step 업데이트: delta = {delta}")

    def prob_convergence(self):
        iter_count = 0
        while True:
            delta = self.probability_update()
            iter_count += 1
            if delta < self.theta:
                break
        self.gridWidget.setData(self.V, self.fixed)
        self.gridWidget.clearPolicy()
        print(f"확률 기반 수렴 완료: {iter_count}회, delta = {delta}")

    def show_policy(self):
        policy = self.compute_policy()
        self.gridWidget.setPolicy(policy)
        print("정책 표시 완료.")


# -------------------------------------------------
# 메인 실행부
# -------------------------------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(800, 800)
    window.show()
    sys.exit(app.exec_())
