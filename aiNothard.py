import importlib
import importlib.util
import logging
import inspect
import math
import os
import re
import json
import copy
import random
import subprocess
import argparse
from collections import deque
from typing import Optional
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("ainothard")
def _ensure(pkg: str, import_name: Optional[str] = None):
    check = import_name or pkg
    if not importlib.util.find_spec(check):
        logger.info(f"Đang cài '{pkg}'...")
        result = subprocess.run(
            ["pip", "install", pkg, "-q"],
            capture_output=True
        )
        if result.returncode != 0:
            logger.error(f"Không thể cài '{pkg}': {result.stderr.decode().strip()}")
_ensure("torch")
_ensure("transformers")
_ensure("peft")
_ensure("opencv-python", "cv2")
_ensure("Pillow", "PIL")
_ensure("diffusers")
_ensure("numpy")
_ensure("scipy")
_ensure("datasets")
_ensure("evaluate")
_ensure("pandas")
_ensure("scikit-learn", "sklearn")
_ensure("joblib")
_ensure("seaborn")
_ensure("matplotlib")
_ensure("torchvision")
_ensure("requests")
from embedded import List, ModelExporter, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2
import requests
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import evaluate
def get_source_code():
    print("""if you need source code you can visit: https://github.com/billciper86/AInotHard , and You can give feedback to help the library improve via email, and you can request to join the GitHub project: quocdat16610@gmail.com""")
    choices = input("Do you want to download the source code as well? (y/n): ").strip().lower()
    if choices =='y':
        url = ""
class Safeinstall:
    @staticmethod
    def ensure(pkg: str, import_name: Optional[str] = None):
        _ensure(pkg, import_name)
class ainothard:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")

    @staticmethod
    def __version__():
        return "1.0.0"

    def build_llm(
        self,
        vocab_size: int = 50000,
        embed_size: int = 512,
        num_heads: int = 8,
        hidden_dim: int = 2048,
        num_layers: int = 6,
        max_seq_length: int = 512,
    ):
        model = TransformerModel(
            vocab_size=vocab_size,
            embed_size=embed_size,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_seq_length=max_seq_length,
        ).to(self.device)
        return model
    def create_model(
        self,
        vocab_size: int = 50000,
        embed_size: int = 512,
        num_heads: int = 8,
        hidden_dim: int = 2048,
        num_layers: int = 6,
        max_seq_length: int = 512,
    ):
        return TransformerModel(
            vocab_size=vocab_size,
            embed_size=embed_size,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_seq_length=max_seq_length,
        ).to(self.device)
class ToolLLM:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        torch_dtype = torch.float16 if torch.cuda.is_available() and device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype
        ).to(device=self.device)

    def sort_data_local(self, folder_path: str) -> str:
        """Đọc tất cả file .txt trong folder và ghép lại."""
        combined_text = ""
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                
                with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                    combined_text += f.read() + "\n"
        return combined_text

    def sort_data_url(self, url: str) -> str:
        """Tải nội dung từ URL và làm sạch HTML tag."""
        def clean_data(data: str) -> str:
            data = re.sub(r'<.*?>', '', data)
            data = re.sub(r'\s+', ' ', data)
            data = data.strip().lower()
            return data

        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return clean_data(response.text)

    def pre_training(
        self,
        text_data: str,
        vocab_size: int,
        batch_size: int,
        sequence_length: int,
        so_du: int,
    ) -> torch.Tensor:
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        max_tokens_needed = batch_size * sequence_length * so_du
        encoded_input = self.tokenizer(
            text_data,
            truncation=True,
            max_length=max_tokens_needed,
            return_tensors="pt",
        )
        input_ids = encoded_input["input_ids"].squeeze(0)
        total_tokens = input_ids.size(0)
        num_chunks = total_tokens // sequence_length

        if num_chunks == 0:
            raise ValueError("Data quá ngắn, không đủ để tạo 1 sequence.")

        input_ids = input_ids[: num_chunks * sequence_length]
        all_batches = input_ids.view(num_chunks, sequence_length)

        if all_batches.size(0) >= batch_size:
            data = all_batches[:batch_size, :]
        else:
            logger.warning(f"Dữ liệu chỉ đủ tạo {all_batches.size(0)} batch.")
            data = all_batches

        return data.to(self.device)

    def train_loop(
        self,
        data: torch.Tensor,
        epochs: int = 3,
        learning_rate: float = 5e-5,
    ):
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        logger.info(
            f"Bắt đầu train {epochs} epoch | batch={data.size(0)} | seq={data.size(1)} | device={self.device}"
        )
        for epoch in range(epochs):
            optimizer.zero_grad()
            if hasattr(self.model, "fc_out") or isinstance(self.model, nn.DataParallel):
                logits = self.model(data)
                loss = criterion(logits.view(-1, logits.size(-1)), data.view(-1))
            else:
                outputs = self.model(input_ids=data, labels=data)
                loss = outputs.loss
            loss.backward()
            optimizer.step()
            logger.info(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f}")

        if self.device == "cuda":
            torch.cuda.empty_cache()

    def evaluate_model_DorQ(self, y_true, y_pred):
        y_true_tensor = torch.tensor(y_true, dtype=torch.float32).to(self.device)
        y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32).to(self.device)
        mse = F.mse_loss(y_pred_tensor, y_true_tensor).item()
        mae = F.l1_loss(y_pred_tensor, y_true_tensor).item()
        return mse, mae

    def predict(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True,
        skip_special_tokens: bool = True,
    ) -> str:
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=skip_special_tokens)

    def save_model(self, file_path: str):
        self.model.save_pretrained(file_path)
        self.tokenizer.save_pretrained(file_path)
        logger.info(f"Model và tokenizer đã lưu tại: {file_path}")

    def load_model(self, file_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(file_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            file_path, torch_dtype=torch.float16
        ).to(self.device)
        logger.info(f"Model và tokenizer đã tải từ: {file_path}")
class Q_Learning:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        neurons: int,
        device: str = "cpu",
        gamma: float = 0.9,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Linear(state_size, neurons),
            nn.ReLU(),
            nn.Linear(neurons, action_size),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.gamma = gamma

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_from_memory(self, batch_size: int):
        if len(self.memory) < batch_size:
            return None

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.tensor(np.array(states),      dtype=torch.float32).to(self.device)
        actions     = torch.tensor(actions,               dtype=torch.long   ).to(self.device)
        rewards     = torch.tensor(rewards,               dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones       = torch.tensor(dones,                 dtype=torch.float32).to(self.device)

        q_values  = self.model(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            max_next_q = self.model(next_states).max(1)[0]
            target_q   = rewards + (1 - dones) * self.gamma * max_next_q

        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()  # LỖI GỐC: thiếu return loss.item()

    def act(self, state, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.model[-1].out_features - 1)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.argmax().item()

    def save_model(self, file_path: str):
        torch.save(self.model.state_dict(), file_path)

    def load_model(self, file_path: str):
        self.model.load_state_dict(torch.load(file_path, map_location=self.device))
        self.model.eval()
class machine_learning:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None

    def train_test_split(self, X, y, test_size: float = 0.2, random_state: int = 42):
        np.random.seed(random_state)
        indices  = np.arange(len(X))
        np.random.shuffle(indices)
        split    = int(len(X) * (1 - test_size))
        X_train, X_test = X[indices[:split]], X[indices[split:]]
        y_train, y_test = y[indices[:split]], y[indices[split:]]
        return X_train, X_test, y_train, y_test

    def linear_regression(self, x, y):
        x_t = torch.tensor(x, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).to(self.device)
        x_mean, y_mean = torch.mean(x_t), torch.mean(y_t)
        slope     = torch.sum((x_t - x_mean) * (y_t - y_mean)) / torch.sum((x_t - x_mean) ** 2)
        intercept = y_mean - slope * x_mean
        return slope.item(), intercept.item()

    def gradient_descent(self, x, y, learning_rate: float = 0.1, epochs: int = 1000):
        x_t = torch.tensor(x, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).to(self.device)
        slope     = torch.randn(1, requires_grad=True, device=self.device)
        intercept = torch.randn(1, requires_grad=True, device=self.device)
        for epoch in range(epochs):
            y_pred = slope * x_t + intercept
            loss   = F.mse_loss(y_pred, y_t)
            loss.backward()
            with torch.no_grad():
                slope     -= learning_rate * slope.grad
                intercept -= learning_rate * intercept.grad
                slope.grad.zero_()
                intercept.grad.zero_()
            if (epoch + 1) % 100 == 0:
                logger.info(f"Epoch {epoch + 1} | Loss: {loss.item():.4f}")
        return slope.item(), intercept.item()

    def evaluate_model_DorQ(self, y_true, y_pred):
        y_true_t = torch.tensor(y_true, dtype=torch.float32).to(self.device)
        y_pred_t = torch.tensor(y_pred, dtype=torch.float32).to(self.device)
        return F.mse_loss(y_pred_t, y_true_t).item(), F.l1_loss(y_pred_t, y_true_t).item()

    def knn(self, X_train, y_train, X_test, k: int = 3):
        X_tr = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        X_te = torch.tensor(X_test,  dtype=torch.float32).to(self.device)
        y_tr = torch.tensor(y_train, dtype=torch.long   ).to(self.device)
        preds = []
        for x in X_te:
            distances = torch.norm(X_tr - x, dim=1)
            indices   = torch.topk(distances, k, largest=False).indices
            preds.append(torch.mode(y_tr[indices]).values.item())
        return np.array(preds)

    def naive_bayes(self, X_train, y_train, X_test):
        X_tr = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        X_te = torch.tensor(X_test,  dtype=torch.float32).to(self.device)
        y_tr = torch.tensor(y_train, dtype=torch.long   ).to(self.device)
        classes        = torch.unique(y_tr)
        class_priors   = {c.item(): (y_tr == c).float().mean().item() for c in classes}
        likelihoods    = {}
        for c in classes:
            cd = X_tr[y_tr == c]
            likelihoods[c.item()] = {"mean": cd.mean(0), "std": cd.std(0) + 1e-6}
        preds = []
        for x in X_te:
            probs = {}
            for c in classes:
                m, s  = likelihoods[c.item()]["mean"], likelihoods[c.item()]["std"]
                prob  = torch.prod(1 / (s * np.sqrt(2 * np.pi)) * torch.exp(-0.5 * ((x - m) / s) ** 2))
                probs[c.item()] = class_priors[c.item()] * prob.item()
            preds.append(max(probs, key=probs.get))
        return np.array(preds)

    def decision_tree(self, X_train, y_train, X_test, max_depth: int = 5):
        self.model = DecisionTreeClassifier(max_depth=max_depth)
        self.model.fit(X_train, y_train)
        return self.model.predict(X_test)

    def plot_decision_tree(self, feature_names=None, class_names=None):
        plt.figure(figsize=(20, 10))
        plot_tree(self.model, feature_names=feature_names, class_names=class_names,
                  filled=True, rounded=True)
        plt.show()

    def plot_results(self, X_test, y_test, y_pred, target_names=None):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=target_names, yticklabels=target_names)
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
        plt.show()
        if isinstance(self.model, DecisionTreeClassifier):
            self.plot_decision_tree()

    def random_forest(self, X_train, y_train, X_test, n_estimators: int = 100, max_depth: int = 5):
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)
        return model.predict(X_test)

    def svm(self, X_train, y_train, X_test, kernel: str = "rbf"):
        model = SVC(kernel=kernel)
        model.fit(X_train, y_train)
        return model.predict(X_test)

    def evaluate_classification(self, y_true, y_pred):
        return (
            accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, average="weighted", zero_division=0),
            recall_score(y_true, y_pred, average="weighted", zero_division=0),
            f1_score(y_true, y_pred, average="weighted", zero_division=0),
        )

    def save_model(self, file_path: str):
        if self.model is None:
            logger.warning("Không có model để lưu.")
            return
        joblib.dump(self.model, file_path)
        logger.info(f"Model đã lưu tại: {file_path}")

    def load_model(self, file_path: str):
        if not os.path.exists(file_path):
            logger.error(f"Không tìm thấy model tại: {file_path}")
            return
        self.model = joblib.load(file_path)
        logger.info(f"Model đã tải từ: {file_path}")
class deep_q_learning:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        neurons: int,
        device: str = "cpu",
        gamma: float = 0.9,
        update_target_steps: int = 5,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Linear(state_size, neurons), nn.ReLU(), nn.Linear(neurons, action_size)
        ).to(self.device)
        self.optimizer         = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.memory            = deque(maxlen=10000)
        self.gamma             = gamma
        self.target_model      = copy.deepcopy(self.model).to(self.device)
        self.target_model.eval()
        self.update_target_steps = update_target_steps
        self.train_step        = 0

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_from_memory(self, batch_size: int):
        if len(self.memory) < batch_size:
            return None

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.tensor(np.array(states),      dtype=torch.float32).to(self.device)
        actions     = torch.tensor(actions,               dtype=torch.long   ).to(self.device)
        rewards     = torch.tensor(rewards,               dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones       = torch.tensor(dones,                 dtype=torch.float32).to(self.device)

        q_values  = self.model(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            max_next_q = self.target_model(next_states).max(1)[0]
            target_q   = rewards + (1 - dones) * self.gamma * max_next_q

        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step += 1
        if self.train_step % self.update_target_steps == 0:
            self.update_target_network()
        return loss.item()

    def act(self, state, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.model[-1].out_features - 1)
        self.model.eval()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        self.model.train()
        return q_values.argmax().item()

    def save_model(self, file_path: str):
        torch.save(self.model.state_dict(), file_path)

    def load_model(self, file_path: str):
        self.model.load_state_dict(torch.load(file_path, map_location=self.device))
        self.model.eval()

    def train_loop(
        self,
        env,
        episodes: int = 1000,
        max_steps: int = 200,
        batch_size: int = 64,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        epsilon = epsilon_start
        for episode in range(episodes):
            reset_result = env.reset()
            state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            total_reward = 0

            for _ in range(max_steps):
                action = self.act(state, epsilon)
                # LỖI GỐC: gym mới trả về 5 giá trị (thêm truncated), cũ trả về 4
                step_result = env.step(action)
                if len(step_result) == 5:
                    next_state, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    next_state, reward, done, _ = step_result

                self.remember(state, action, reward, next_state, done)
                state        = next_state
                total_reward += reward
                if done:
                    break

            self.train_from_memory(batch_size)
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            logger.info(f"Episode {episode + 1}/{episodes} | Reward: {total_reward} | ε: {epsilon:.4f}")
class deep_learning:
    def __init__(
        self,
        device: str = "cpu",
        input_size: int = 784,
        hidden_size: int = 128,
        output_size: int = 10,
    ):
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def train_loop(self, train_loader, epochs: int = 10):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(batch_x), batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            logger.info(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(train_loader):.4f}")

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32).to(self.device)
            return torch.argmax(self.model(x_t), dim=1).cpu().numpy()

    def save_model(self, file_path: str):
        torch.save(self.model.state_dict(), file_path)

    def load_model(self, file_path: str):
        self.model.load_state_dict(torch.load(file_path, map_location=self.device))
        self.model.eval()
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1    = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(planes)
        self.conv2    = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2      = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)
class CNN_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes: int = 512):
        super().__init__()
        self.in_planes = 64
        self.conv1  = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        return self.linear(out.view(out.size(0), -1))

    def train_cnn(self, model, train_loader, device: str = "cpu", epochs: int = 10):
        model.train(); model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward(); optimizer.step()
                total_loss += loss.item()
            logger.info(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(train_loader):.4f}")

    def save_model(self, file_path: str):
        torch.save(self.state_dict(), file_path)
        logger.info(f"Model đã lưu tại: {file_path}")
        
    def load_model(self, file_path: str, device: str = "cpu"):
        state_dict = torch.load(file_path, map_location=torch.device(device))
        self.load_state_dict(state_dict)
        self.to(device); self.eval()
        logger.info(f"Model đã tải từ: {file_path} | device: {device}")
class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int,
        max_seq_length: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embed_size)
        self.embed_scale = math.sqrt(embed_size)
        self.dropout    = nn.Dropout(dropout)
        self.register_buffer(
            "pos_encoding",
            self._generate_positional_encoding(max_seq_length, embed_size),
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm   = nn.LayerNorm(embed_size)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.fc_out.weight = self.embedding.weight
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if self.fc_out.bias is not None:
            nn.init.zeros_(self.fc_out.bias)

    def _generate_positional_encoding(self, max_seq_length: int, embed_size: int) -> torch.Tensor:
        pe       = torch.zeros(max_seq_length, embed_size)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _generate_causal_mask(self, seq_length: int, device: torch.device) -> torch.Tensor:
        """Tam giác trên -inf: token i chỉ attend token 0..i-1."""
        return torch.triu(
            torch.full((seq_length, seq_length), float("-inf"), device=device), diagonal=1
        )

    def forward(self, input_ids: torch.Tensor = None, **kwargs) -> torch.Tensor:
        seq_length   = input_ids.size(1)
        x            = self.embedding(input_ids) * self.embed_scale
        x            = x + self.pos_encoding[:, :seq_length, :]
        x            = self.dropout(x)
        causal_mask  = self._generate_causal_mask(seq_length, input_ids.device)
        out = self.transformer_decoder(
            tgt=x, memory=x,
            tgt_mask=causal_mask,
            tgt_is_causal=True,
        )
        return self.fc_out(self.norm(out))

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int = None,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits           = self(input_ids)[:, -1, :]
                logits           = logits / max(temperature, 1e-8)
                if top_k > 0:
                    kth          = torch.topk(logits, top_k).values[:, -1, None]
                    logits       = logits.masked_fill(logits < kth, float("-inf"))
                if top_p < 1.0:
                    sorted_l, idx = torch.sort(logits, descending=True)
                    cum          = torch.cumsum(torch.softmax(sorted_l, -1), -1)
                    remove       = cum - torch.softmax(sorted_l, -1) > top_p
                    remove[:, 1:] = remove[:, :-1].clone(); remove[:, 0] = False
                    logits       = logits.masked_fill(remove.scatter(1, idx, remove), float("-inf"))
                next_token       = torch.multinomial(torch.softmax(logits, -1), 1)
                input_ids        = torch.cat([input_ids, next_token], dim=1)
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break
        return input_ids

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_model(self, file_path: str):
        torch.save(self.state_dict(), file_path)
        logger.info(f"Model đã lưu tại: {file_path} ({self.count_parameters():,} params)")

    def load_model(self, file_path: str, device: str = "cpu"):
        self.load_state_dict(torch.load(file_path, map_location=torch.device(device)))
        self.to(device); self.eval()
        logger.info(f"Model đã tải từ: {file_path} | device: {device}")
class action_recognition:
    def __init__(self, device: str = "cpu"):
        self.device = device
        logger.warning("action_recognition chưa được implement.")
class videofolderdataset(Dataset):
    def __init__(self, folder_path: str, frames_per_video: int = 16, transform=None):
        self.folder_path      = folder_path
        self.frames_per_video = frames_per_video
        self.transform        = transform
        self.video_files      = [
            f for f in os.listdir(folder_path)
            if f.endswith((".mp4", ".avi", ".mov"))
        ]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        path   = os.path.join(self.folder_path, self.video_files[idx])
        frames = self._extract_frames(path)
        if self.transform:
            frames = torch.stack([self.transform(f) for f in frames])
        return frames

    def _extract_frames(self, video_path: str):
        cap          = cv2.VideoCapture(video_path)
        total        = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices      = np.linspace(0, max(total - 1, 0), self.frames_per_video, dtype=int)
        frames       = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            else:
                frames.append(Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)))
        cap.release()
        return frames
class learnbyvideo:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")

    def prepare_data(self, folder_path: str, batch_size: int = 4) -> DataLoader:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return DataLoader(
            videofolderdataset(folder_path, transform=transform),
            batch_size=batch_size, shuffle=True,
        )

    def train_on_folder(self, folder_path: str, epochs: int = 10, batch_size: int = 4):
        dataloader = self.prepare_data(folder_path, batch_size)
        for epoch in range(epochs):
            for batch in dataloader:
                batch = batch.to(self.device)
                logger.info(f"Epoch {epoch + 1} | Batch size: {batch.size(0)}")

    def train_on_url(self, video_url: str, epochs: int = 10):
        response   = requests.get(video_url, timeout=30)
        video_path = "temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(response.content)
        folder = os.path.dirname(os.path.abspath(video_path))
        self.train_on_folder(folder, epochs=epochs)

    def save_model(self, model, model_name: str = "learnbyvideo.pth"):
        folder = os.path.dirname(model_name)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(model.state_dict(), model_name)
        logger.info(f"Model đã lưu tại: {model_name}")

    def load_model(self, model, path: str = "learnbyvideo.pth"):
        if not os.path.exists(path):
            logger.error(f"Không tìm thấy model tại: {path}")
            return
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device); model.eval()
        logger.info(f"Model đã tải từ: {path}")
class RLHF:
    """
    Reinforcement Learning from Human Feedback tích hợp vào aiNothard.
 
    Quy trình gồm 3 giai đoạn:
    ─────────────────────────────────────────────────
    Giai đoạn 1 – Supervised Fine-Tuning (SFT)
        Tinh chỉnh model ngôn ngữ trên tập dữ liệu
        prompt → response chất lượng cao.
 
    Giai đoạn 2 – Reward Model Training
        Huấn luyện mô hình reward dự đoán phản hồi nào
        được con người ưa thích hơn (Bradley-Terry loss).
 
    Giai đoạn 3 – PPO / REINFORCE Fine-tuning
        Dùng reward model để cải thiện policy (LLM chính)
        qua thuật toán REINFORCE đơn giản + KL penalty
        so với SFT reference model.
    ─────────────────────────────────────────────────
 
    Ví dụ sử dụng nhanh:
    >>> ai   = ainothard()
    >>> llm  = ai.build_llm()
    >>> rlhf = RLHF(policy_model=llm, embed_size=512)
    >>> # --- Giai đoạn 1: SFT ---
    >>> sft_data = [{"input": [1,2,3], "target": [2,3,4]}]
    >>> rlhf.sft_train(sft_data, epochs=3)
    >>> # --- Giai đoạn 2: Reward model ---
    >>> pairs = [{"chosen":[1,2,3,4], "rejected":[1,2,5,6]}]
    >>> rlhf.reward_train(pairs, epochs=5)
    >>> # --- Giai đoạn 3: PPO/REINFORCE ---
    >>> prompts = [torch.tensor([1,2,3])]
    >>> rlhf.ppo_train(prompts, epochs=3)
    """
 
    def __init__(
        self,
        policy_model:   nn.Module,
        embed_size:     int   = 512,
        reward_hidden:  int   = 256,
        kl_coef:        float = 0.1,
        lr_sft:         float = 5e-5,
        lr_reward:      float = 1e-4,
        lr_ppo:         float = 1e-5,
        device:         str   = "cpu",
    ):
        self.device      = torch.device("cuda" if torch.cuda.is_available() else device)
        self.policy      = policy_model.to(self.device)
        self.ref_policy  = copy.deepcopy(policy_model).to(self.device)   # frozen reference
        for p in self.ref_policy.parameters():
            p.requires_grad_(False)
        self.reward_model = RewardModel(
            copy.deepcopy(policy_model), embed_size, reward_hidden
        ).to(self.device)
 
        self.kl_coef   = kl_coef
        self.opt_sft   = torch.optim.AdamW(self.policy.parameters(),       lr=lr_sft)
        self.opt_rm    = torch.optim.AdamW(self.reward_model.parameters(), lr=lr_reward)
        self.opt_ppo   = torch.optim.AdamW(self.policy.parameters(),       lr=lr_ppo)
        self._log      = logging.getLogger("ainothard.RLHF")

    def sft_train(
        self,
        sft_pairs: list,
        epochs:     int   = 3,
        batch_size: int   = 8,
    ):
        """
        Tinh chỉnh policy bằng dữ liệu (input, target) chất lượng cao.
 
        Parameters
        ----------
        sft_pairs : list of dict {"input": List[int], "target": List[int]}
        epochs    : số epoch huấn luyện
        batch_size: kích thước batch
 
        Returns
        -------
        list[float] – loss từng epoch
        """
        self._log.info(f"[SFT] Bắt đầu | {len(sft_pairs)} mẫu | {epochs} epoch")
        self.policy.train()
        history = []
        for epoch in range(epochs):
            random.shuffle(sft_pairs)
            total_loss, steps = 0.0, 0
            for i in range(0, len(sft_pairs), batch_size):
                chunk = sft_pairs[i : i + batch_size]
                # Ghép input+target thành 1 chuỗi, dùng target làm label
                max_len = max(len(s["input"]) + len(s["target"]) for s in chunk)
                ids     = torch.zeros(len(chunk), max_len, dtype=torch.long).to(self.device)
                labels  = torch.full((len(chunk), max_len), -100, dtype=torch.long).to(self.device)
                for j, s in enumerate(chunk):
                    seq = s["input"] + s["target"]
                    ids[j, :len(seq)] = torch.tensor(seq)
                    # Chỉ tính loss trên phần target
                    tgt_start = len(s["input"])
                    labels[j, tgt_start : tgt_start + len(s["target"])] = torch.tensor(s["target"])
 
                logits = self.policy(ids)                    # (B, seq, vocab)
                loss   = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
                self.opt_sft.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.opt_sft.step()
                total_loss += loss.item(); steps += 1
 
            avg = total_loss / max(steps, 1)
            history.append(avg)
            self._log.info(f"[SFT] Epoch {epoch+1}/{epochs} | Loss: {avg:.4f}")
 
        # Cập nhật reference policy sau SFT
        self.ref_policy = copy.deepcopy(self.policy).to(self.device)
        for p in self.ref_policy.parameters():
            p.requires_grad_(False)
        self._log.info("[SFT] Hoàn thành. Reference policy đã cập nhật.")
        return history

    def reward_train(
        self,
        preference_pairs: list,
        epochs:           int   = 5,
        batch_size:       int   = 16,
        pad_id:           int   = 0,
    ):
        """
        Huấn luyện reward model theo Bradley-Terry loss.
        Mục tiêu: reward(chosen) > reward(rejected).
 
        Parameters
        ----------
        preference_pairs : list of dict {"chosen": List[int], "rejected": List[int]}
        epochs           : số epoch
        batch_size       : kích thước batch
        pad_id           : token dùng để padding
 
        Returns
        -------
        list[float] – loss từng epoch
        """
        self._log.info(f"[RM] Bắt đầu | {len(preference_pairs)} cặp | {epochs} epoch")
        dataset    = PreferenceDataset(preference_pairs)
        loader     = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            collate_fn=lambda b: _pad_collate(b, pad_id),
        )
        self.reward_model.train()
        history = []
        for epoch in range(epochs):
            total_loss, steps = 0.0, 0
            for chosen, rejected in loader:
                chosen   = chosen.to(self.device)
                rejected = rejected.to(self.device)
                r_chosen   = self.reward_model(chosen)      # (B,)
                r_rejected = self.reward_model(rejected)    # (B,)
                # Bradley-Terry: -log σ(r_chosen - r_rejected)
                loss = -F.logsigmoid(r_chosen - r_rejected).mean()
                self.opt_rm.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.reward_model.parameters(), 1.0)
                self.opt_rm.step()
                total_loss += loss.item(); steps += 1
 
            avg = total_loss / max(steps, 1)
            history.append(avg)
            acc = self._reward_accuracy(loader)
            self._log.info(f"[RM] Epoch {epoch+1}/{epochs} | Loss: {avg:.4f} | Acc: {acc:.2%}")
 
        self._log.info("[RM] Hoàn thành huấn luyện reward model.")
        return history
 
    @torch.no_grad()
    def _reward_accuracy(self, loader) -> float:
        """Tỉ lệ batch mà reward(chosen) > reward(rejected)."""
        self.reward_model.eval()
        correct, total = 0, 0
        for chosen, rejected in loader:
            rc = self.reward_model(chosen.to(self.device))
            rr = self.reward_model(rejected.to(self.device))
            correct += (rc > rr).sum().item()
            total   += rc.size(0)
        self.reward_model.train()
        return correct / max(total, 1)
 
    # ──────────────────────────────────────────
    # GIAI ĐOẠN 3 – PPO / REINFORCE
    # ──────────────────────────────────────────
    def ppo_train(
        self,
        prompt_ids:      list,
        epochs:          int   = 3,
        max_new_tokens:  int   = 64,
        temperature:     float = 0.9,
        top_k:           int   = 50,
    ):
        """
        Cải thiện policy bằng REINFORCE + KL divergence penalty.
 
        Phần thưởng:  R_total = reward_model(response) − kl_coef * KL(policy ∥ ref)
 
        Parameters
        ----------
        prompt_ids      : list of torch.Tensor, mỗi phần tử là token ids của 1 prompt
        epochs          : số lần lặp qua toàn bộ prompt
        max_new_tokens  : số token tối đa sinh ra
        temperature     : nhiệt độ sampling
        top_k           : top-k sampling
 
        Returns
        -------
        list[float] – tổng reward trung bình từng epoch
        """
        self._log.info(f"[PPO] Bắt đầu | {len(prompt_ids)} prompt | {epochs} epoch")
        history = []
        for epoch in range(epochs):
            total_reward, steps = 0.0, 0
            for prompt in prompt_ids:
                prompt_t = prompt.unsqueeze(0).to(self.device)  # (1, prompt_len)
 
                # --- Sinh response từ policy ---
                response_ids = self._generate(prompt_t, max_new_tokens, temperature, top_k)
                full_ids     = torch.cat([prompt_t, response_ids], dim=1)
 
                # --- Tính reward ---
                with torch.no_grad():
                    reward = self.reward_model(full_ids).squeeze()
 
                # --- Tính KL penalty ---
                kl = self._kl_divergence(full_ids)
                total_r = reward - self.kl_coef * kl
 
                # --- REINFORCE: tối đa hoá E[R * log π(a|s)] ---
                log_probs = self._log_probs(full_ids)           # (seq-1,)
                loss      = -(total_r.detach() * log_probs.mean())
 
                self.opt_ppo.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.opt_ppo.step()
 
                total_reward += total_r.item(); steps += 1
 
            avg = total_reward / max(steps, 1)
            history.append(avg)
            self._log.info(f"[PPO] Epoch {epoch+1}/{epochs} | Avg Reward: {avg:.4f}")
 
        self._log.info("[PPO] Hoàn thành RLHF fine-tuning.")
        return history
 
    # ──────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────
    @torch.no_grad()
    def _generate(
        self,
        input_ids:     torch.Tensor,
        max_new_tokens: int,
        temperature:   float,
        top_k:         int,
    ) -> torch.Tensor:
        """Sinh response token (không tính gradient)."""
        self.policy.eval()
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            logits    = self.policy(generated)[:, -1, :] / max(temperature, 1e-8)
            if top_k > 0:
                kth   = torch.topk(logits, top_k).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs     = torch.softmax(logits, dim=-1)
            next_tok  = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_tok], dim=1)
        self.policy.train()
        return generated[:, input_ids.size(1):]     # chỉ phần mới sinh
 
    def _log_probs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Log-probability của từng token trong chuỗi."""
        logits    = self.policy(input_ids)           # (1, seq, vocab)
        log_p     = F.log_softmax(logits, dim=-1)
        # token t+1 được dự đoán tại vị trí t
        token_ids = input_ids[:, 1:]                 # (1, seq-1)
        return log_p[:, :-1, :].squeeze(0).gather(1, token_ids.squeeze(0).unsqueeze(1)).squeeze(1)
 
    def _kl_divergence(self, input_ids: torch.Tensor) -> torch.Tensor:
        """KL(policy ∥ ref_policy) trung bình trên toàn chuỗi."""
        with torch.no_grad():
            ref_logits = self.ref_policy(input_ids)
            ref_log_p  = F.log_softmax(ref_logits, dim=-1)
        logits  = self.policy(input_ids)
        log_p   = F.log_softmax(logits, dim=-1)
        p       = torch.exp(log_p)
        kl      = (p * (log_p - ref_log_p)).sum(dim=-1).mean()
        return kl
 
    # ──────────────────────────────────────────
    # TIỆN ÍCH: đánh giá & lưu/tải
    # ──────────────────────────────────────────
    @torch.no_grad()
    def score_response(self, input_ids: torch.Tensor) -> float:
        """Trả về điểm reward cho 1 chuỗi token (dùng để kiểm tra nhanh)."""
        self.reward_model.eval()
        ids    = input_ids.unsqueeze(0).to(self.device) if input_ids.dim() == 1 else input_ids.to(self.device)
        score  = self.reward_model(ids).item()
        self.reward_model.train()
        return score
 
    def save(self, policy_path: str, reward_path: str):
        """Lưu policy và reward model."""
        torch.save(self.policy.state_dict(),       policy_path)
        torch.save(self.reward_model.state_dict(), reward_path)
        self._log.info(f"[RLHF] Đã lưu policy → {policy_path} | reward → {reward_path}")
 
    def load(self, policy_path: str, reward_path: str):
        """Tải policy và reward model từ file."""
        self.policy.load_state_dict(torch.load(policy_path,  map_location=self.device))
        self.reward_model.load_state_dict(torch.load(reward_path, map_location=self.device))
        self.ref_policy = copy.deepcopy(self.policy).to(self.device)
        for p in self.ref_policy.parameters():
            p.requires_grad_(False)
        self._log.info(f"[RLHF] Đã tải policy ← {policy_path} | reward ← {reward_path}")
class RobotDeployer:
    def __init__(self, model_instance):
        if hasattr(model_instance, 'model'):
            self.model = model_instance.model
        else:
            self.model = model_instance
            
        self.exporter = ModelExporter(self.model)

    def from_pth(self, pth_path: str):
        try:
            self.model.load_state_dict(torch.load(pth_path))
            print(f"Successfully loaded weights from {pth_path}")
        except Exception as e:
            print(f"Error loading .pth file: {e}")
        return self

    def export_for_arduino(self, filename: str, model_name: str = "RobotBrain"):
        self.exporter.export_arduino_h(filename, model_name=model_name)
        print(f"==> Created file: {filename}")

    def export_for_esp32(self, filename: str, model_name: str = "RobotBrain"):
        self.exporter.export_c_header(filename, model_name=model_name)
        print(f"==> Created file: {filename}")
class DataScaler:
    def __init__(self, features_range: List[Tuple[float, float]]):
        self.ranges = features_range

    def scale(self, raw_data: List[float]) -> np.ndarray:
        scaled = []
        for i, val in enumerate(raw_data):
            mi, ma = self.ranges[i]
            #(val - min) / (max - min)
            s = (val - mi) / (ma - mi + 1e-6) 
            scaled.append(max(0.0, min(1.0, s)))
        return np.array([scaled], dtype=np.float32)

    def generate_cpp_macro(self):
        cpp_code = "// --- Copy vao Arduino de chuan hoa du lieu ---\n"
        for i, (mi, ma) in enumerate(self.ranges):
            cpp_code += f"#define SENSOR_{i}_MIN {mi}\n"
            cpp_code += f"#define SENSOR_{i}_MAX {ma}\n"
        return cpp_code
class CPlusPlusBridge:
    def __init__(self, model_instance):
        if hasattr(model_instance, 'model'):
            self.model = model_instance.model
        else:
            self.model = model_instance
            
    def generate_inference_code(self, model_name="RobotQ"):
        """Tạo hàm logic C++ tương ứng với cấu trúc mạng Neural"""
        layers = [m for m in self.model if isinstance(m, nn.Linear)]
        
        input_size = layers[0].in_features
        hidden_size = layers[0].out_features
        output_size = layers[-1].out_features

        cpp_template = f"""
/* AUTO-GENERATED BY aiNothard CPlusPlusBridge
   Hàm thực thi AI trực tiếp trên vi điều khiển
*/
#include "{model_name}.h"

int {model_name}_predict(float* input_data) {{
    float hidden[{hidden_size}];
    float output[{output_size}];

    // --- LAYER 1: INPUT -> HIDDEN (Linear + ReLU) ---
    for (int i = 0; i < {hidden_size}; i++) {{
        hidden[i] = pgm_read_float(&{model_name}_0_bias[i]);
        for (int j = 0; j < {input_size}; j++) {{
            hidden[i] += input_data[j] * pgm_read_float(&{model_name}_0_weight[i * {input_size} + j]);
        }}
        if (hidden[i] < 0) hidden[i] = 0; // ReLU
    }}

    // --- LAYER 2: HIDDEN -> OUTPUT (Linear) ---
    int best_action = 0;
    float max_val = -9999.0;
    for (int i = 0; i < {output_size}; i++) {{
        output[i] = pgm_read_float(&{model_name}_2_bias[i]);
        for (int j = 0; j < {hidden_size}; j++) {{
            output[i] += hidden[j] * pgm_read_float(&{model_name}_2_weight[i * {hidden_size} + j]);
        }}
        // Argmax: Tim hanh dong co gia tri cao nhat
        if (output[i] > max_val) {{
            max_val = output[i];
            best_action = i;
        }}
    }}
    return best_action;
}}
"""
        return cpp_template
class MemoryChat:
    def __init__(self, model, tokenizer, max_tokens=512, strategy="sliding"):
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.strategy = strategy 
        self.history = deque(maxlen=20)
        self.long_term = []
    
    def chat(self, user_input: str) -> str:
        self.history.append(f"User: {user_input}")
        context = "\n".join(self.long_term + list(self.history))
        
        # Tokenize & truncate nếu quá dài
        tokens = self.tokenizer(context, return_tensors="pt", 
                                truncation=True, max_length=self.max_tokens)
        
        output = self.model.generate(**tokens, max_new_tokens=200)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        self.history.append(f"AI: {response}")
        return response
    
    def summarize_and_compress(self):
        """Gọi khi history quá dài"""
        old = "\n".join(list(self.history)[:10])
        # Dùng model tóm tắt
        summary_input = self.tokenizer(
            f"Tóm tắt:\n{old}", return_tensors="pt", truncation=True
        )
        out = self.model.generate(**summary_input, max_new_tokens=100)
        summary = self.tokenizer.decode(out[0], skip_special_tokens=True)
        
        self.long_term.append(f"[Tóm tắt]: {summary}")
        # Xóa 10 tin cũ
        for _ in range(10):
            if self.history:
                self.history.popleft()
