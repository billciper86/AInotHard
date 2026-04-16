import importlib 
import math
import os
import cv2
from matplotlib import transforms
if not importlib.util.find_spec("torch"):
    os.system("pip install torch")
if not importlib.util.find_spec("transformers"):
    os.system("pip install transformers")
if not importlib.util.find_spec("peft"):
    os.system("pip install peft")
if not importlib.util.find_spec("cv2"):
    os.system("pip install opencv-python")
if not importlib.util.find_spec("PIL"):
    os.system("pip install Pillow")
if not importlib.util.find_spec("diffusers"):
    os.system("pip install diffusers")
if not importlib.util.find_spec("numpy"):
    os.system("pip install numpy")
if not importlib.util.find_spec("scipy"):
    os.system("pip install scipy")
if not importlib.util.find_spec("datasets"):
    os.system("pip install datasets")
if not importlib.util.find_spec("evaluate"):
    os.system("pip install evaluate")
if not importlib.util.find_spec("pandas"):
    os.system("pip install pandas")
if not importlib.util.find_spec("sklearn"):
    os.system("pip install scikit-learn")
if not importlib.util.find_spec("joblib"):
    os.system("pip install joblib")
from attrs import inspect
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from PIL import Image
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch.nn as nn
import re
import requests
import json
from torchvision import transforms
import evaluate
import random
import pandas as pd
from collections import deque
import copy
from torch.utils.data import Dataset, DataLoader
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import seaborn as sns
from sklearn.metrics import confusion_matrix
class ainothard():
    def __init__(self, device: str):
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
    @staticmethod
    def __version__():
            return "1.1.0"
    def build_llm(self, vocab_size: int = 50000, embed_size: int = 512, num_heads: int = 8, hidden_dim: int = 2048, num_layers: int = 6, max_seq_length: int = 512):
        model = Tranformermodel(
            vocab_size = vocab_size,
            embed_size = embed_size,
            num_heads = num_heads,
            hidden_dim = hidden_dim,
            num_layers = num_layers,
            max_seq_length = max_seq_length
        ).to(self.device)
        return model
    def create_model(self, vocab_size: int = 50000, embed_size: int = 512, num_heads: int = 8, hidden_dim: int = 2048, num_layers: int = 6, max_seq_length: int = 512):
        return Tranformermodel(
            vocab_size = vocab_size,
            embed_size = embed_size,
            num_heads = num_heads,
            hidden_dim = hidden_dim,
            num_layers = num_layers,
            max_seq_length = max_seq_length).to(self.device)
class ToolLLM():
    def __init__(self, model_name: str, device: str, mode_train):
        self.model_name = model_name
        self.device = device
        self.mode_train = mode_train
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    def sort_data_local(self, folder_path: str ):
        combined_text = ""
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                    combined_text += file.read() + "\n"
        return combined_text
    
    def sort_data_url(self, url: str, ):
        def clean_data(data):
            data = re.sub(r'[r<.*?>]', '', data)
            data = re.sub(r'[\s+]', ' ', data)
            data = data.strip().lower()
            return data
        response = requests.get(url)
        cleaned_data = clean_data(response.text)
        return cleaned_data
    def pre_training(self, text_data: str, vocab_size: int, batch_size: int , sequence_length: int, so_du: int):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        max_tokens_needed = batch_size * sequence_length * so_du
        encoded_input = self.tokenizer(
            text_data,
            truncation= True,
            max_length=max_tokens_needed,
            return_tensors="pt"
        )
        input_ids = encoded_input["input_ids"].squeeze(0)
        total_tokens = input_ids.size(0)
        num_chunks = total_tokens // sequence_length
        if num_chunks == 0:
            raise ValueError("data is too short, not enough for 1 sequence")
        input_ids = input_ids[:num_chunks * sequence_length]
        all_batches = input_ids.view(num_chunks, sequence_length)
        if all_batches.size(0) >= batch_size:
            data = all_batches[:batch_size, :]
        else:
            print(f"Cảnh báo: Dữ liệu chỉ đủ tạo {all_batches.size(0)} batch.")
            data = all_batches
        data = data.to(self.device)
        return data
    def train_loop(self, data: torch.Tensor, epochs: int = 3, learning_rate: float = 5e-5, device: str = "cuda"):
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        print(f'Starting training for {epochs} epochs with batch size {data.size(0)} and sequence length {data.size(1)} with {self.device}')
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(input_ids=data, labels=data)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        if device == "cuda":
            torch.cuda.empty_cache()
    def evaluate_model_DorQ(self, y_true, y_pred):
        y_true_tensor = torch.tensor(y_true, dtype=torch.float32).to(self.device)
        y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32).to(self.device)
        mse = F.mse_loss(y_pred_tensor, y_true_tensor).item()
        mae = F.l1_loss(y_pred_tensor, y_true_tensor).item()
        return mse, mae
    def predict(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7, do_sample: bool = True, skip_special_tokens: bool = True):
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens= max_new_tokens,do_sample = do_sample, temperature=temperature )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=skip_special_tokens)
    def save_model(self, file_path: str):
        self.model.save_pretrained(file_path)
        self.tokenizer.save_pretrained(file_path)
        print(f"Model and tokenizer saved to {file_path}")
    def load_model(self, file_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(file_path)
        self.model = AutoModelForCausalLM.from_pretrained(file_path, torch_dtype=torch.float16).to(self.device)
        print(f'Model and tokenizer loaded from {file_path}')
class Q_Learning():
    def __init__(self, state_size: int, action_size: int,neurons: int, device: str, gamma: float = 0.9):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Linear(state_size, neurons),
            nn.ReLU(),
            nn.Linear(neurons, action_size)
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.001)
        self.memory = deque(maxlen=10000)
        self.gamma = gamma
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def train_from_memory(self, batch_size: int):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states= torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        q_values = self.model(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_next_q = self.model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def act(self, state, epsilon: float):
        if random.random()< epsilon:
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
class machine_learning():
    def __init__(self, device: str):
        self.device = device
        self.model = None
    def train_test_split(self, X, y, test_size: float = 0.2, random_state: int = 42):
        np.random.seed(random_state)
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[indices[:split_idx]], X[indices[split_idx:]]
        y_train, y_test = y[indices[:split_idx]], y[indices[split_idx:]]
        return X_train, X_test, y_train, y_test
    def linear_regression(self, x, y):
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        x_mean = torch.mean(x_tensor)
        y_mean = torch.mean(y_tensor)
        numerator = torch.sum((x_tensor - x_mean) * (y_tensor - y_mean))
        denominator = torch.sum((x_tensor - x_mean) ** 2)
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        return slope.item(), intercept.item()
    def gradient_descent(self, x , y, learning_rate: float = 0.1, epochs: int = 1000):
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        slope = torch.randn(1, requires_grad=True, device=self.device)
        intercept = torch.randn(1, requires_grad=True, device=self.device)
        for epoch in range(epochs):
            y_pred = slope * x_tensor + intercept
            loss = F.mse_loss(y_pred, y_tensor)
            loss.backward()
            with torch.no_grad():
                slope -= learning_rate * slope.grad
                intercept -= learning_rate * intercept.grad
                slope.grad.zero_()
                intercept.grad.zero_()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
        return slope.item(), intercept.item()
    def evaluate_model_DorQ(self, y_true, y_pred):
        y_true_tensor = torch.tensor(y_true, dtype=torch.float32).to(self.device)
        y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32).to(self.device)
        mse = F.mse_loss(y_pred_tensor, y_true_tensor).item()
        mae = F.l1_loss(y_pred_tensor, y_true_tensor).item()
        return mse, mae
    def knn(self, X_train, y_train, X_test, k: int = 3):
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        x_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)
        predictions = []
        for x in x_test_tensor:
            distances = torch.norm(X_train_tensor - x, dim=1)
            knn_indices = torch.topk(distances, k, largest=False).indices
            knn_labels = y_train_tensor[knn_indices]
            predicted_label = torch.mode(knn_labels).values.item()
            predictions.append(predicted_label)
        return np.array(predictions)
    def naive_bayes(self, X_train, y_train, X_test):
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)
        classes = torch.unique(y_train_tensor)
        class_priors = {c.item(): (y_train_tensor == c).float().mean().item() for c in classes}
        feature_likelihoods = {}
        for c in classes:
            class_data = X_train_tensor[y_train_tensor == c]
            feature_likelihoods[c.item()] = {
                'mean': class_data.mean(dim=0),
                'std': class_data.std(dim=0) + 1e-6
            }
        predictions = []
        for x in X_test_tensor:
            class_probs = {}
            for c in classes:
                mean = feature_likelihoods[c.item()]['mean']
                std = feature_likelihoods[c.item()]['std']
                likelihood = torch.prod(1 / (std * np.sqrt(2 * np.pi)) * torch.exp(-0.5 * ((x - mean) / std) ** 2))
                class_probs[c.item()] = class_priors[c.item()] * likelihood.item()
            predicted_class = max(class_probs, key=class_probs.get)
            predictions.append(predicted_class)
        return np.array(predictions)
    def decision_tree(self, X_train, y_train, X_test, max_depth: int = 5):
        self.model = DecisionTreeClassifier(max_depth=max_depth)
        self.model.fit(X_train, y_train)
        return self.model.predict(X_test)
    def plot_decision_tree(self, feature_names=None, class_names=None):
        plt.figure(figsize=(20,10))
        plot_tree(self.model, 
                feature_names=feature_names, 
                class_names=class_names, 
                filled=True, 
                rounded=True)
        plt.show()
    def plot_results(self, X_test, y_test, y_pred, target_names=None):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
        if isinstance(self.model, DecisionTreeClassifier):
            self.plot_decision_tree()
    def random_forest(self, X_train, y_train, X_test, n_estimators: int = 100, max_depth: int = 5):
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)
        return model.predict(X_test)
    def svm(self, X_train, y_train, X_test, kernel: str = 'rbf'):
        model = SVC(kernel=kernel)
        model.fit(X_train, y_train)
        return model.predict(X_test)
    def evaluate_classification(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        return accuracy, precision, recall, f1
    def save_model(self, file_path: str):
        if self.model is None:
            print("No model to save.")
            return
        joblib.dump(self.model, file_path)
        print(f'Model saved to {file_path}')
    def load_model(self, file_path: str):
        if not os.path.exists(file_path):
            print(f"No model found at {file_path}")
            return
        self.model = joblib.load(file_path)
        print(f'Model loaded from {file_path}')
class deep_q_learning():
    def __init__(self, state_size: int, action_size: int, neurons: int, device: str, gamma: float = 0.9, update_target_steps: int = 5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Linear(state_size, neurons),
            nn.ReLU(),
            nn.Linear(neurons, action_size)
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.001)
        self.memory = deque(maxlen=10000)
        self.gamma = gamma
        self.target_model = copy.deepcopy(self.model).to(self.device)
        self.target_model.eval()
        self.update_target_steps = update_target_steps
        self.train_step = 0
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def train_from_memory(self, batch_size: int):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states= torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        q_values = self.model(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.train_step += 1
        if self.train_step % self.update_target_steps == 0:
            self.update_target_network()
        return loss.item()
    def act(self, state, epsilon: float):
        if random.random()< epsilon:
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
    def train_loop(self, env, episodes: int = 1000, max_steps: int = 200, batch_size: int = 64, epsilon_start: float = 1.0, epsilon_end: float = 0.01, epsilon_decay: float = 0.995):
        epsilon = epsilon_start
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            for step in range(max_steps):
                action = self.act(state, epsilon)
                next_state, reward, done, _ = env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    break
            self.train_from_memory(batch_size)
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")
class deep_learning():
    def __init__(self, device: str, input_size: int, hidden_size: int, output_size: int):
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss() # Thường dùng cho phân loại

    def train_loop(self, train_loader, epochs: int = 10):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
            outputs = self.model(x_tensor)
            return torch.argmax(outputs, dim=1).cpu().numpy()
    def save_model(self, file_path: str):
        torch.save(self.model.state_dict(), file_path)
    def load_model(self, file_path: str):
        self.model.load_state_dict(torch.load(file_path, map_location=self.device))
        self.model.eval()

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
         
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes)
                )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class CNN_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=512):
        super(CNN_ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    def train_cnn(self, model, train_loader, device: str, epochs: int = 10):
        model.train()
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    def save_model(self, file_path: str):
        torch.save(self.state_dict(), file_path)
        print(f"Saved model to: {file_path}")

    def load_model(self, file_path: str, device: str = 'cpu'):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(file_path, map_location=torch.device(device))
        self.load_state_dict(state_dict)
        self.to(device)
        self.eval()
        print(f"Loaded model from: {file_path} on device: {device} ---")
class Tranformermodel(nn.Module):
    def __init__(self, vocab_size , embed_size: int, num_heads: int, hidden_dim: int, num_layers: int, max_seq_length: int):
        super(Tranformermodel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.register_buffer('pos_encoding', self._generate_positional_encoding(max_seq_length, embed_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            batch_first= True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)
    def _generate_positional_encoding(self, max_seq_length, embed_size):
        pe = torch.zeros(max_seq_length, embed_size)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        seq_length = x.size(1)
        out = self.embedding(x) + self.pos_encoding[:, :seq_length, :]
        out = self.transformer_encoder(out)
        logits = self.fc_out(out)
        return logits
    def save_model(self, file_path: str):
        torch.save(self.state_dict(), file_path)
        print(f"Saved model to: {file_path}")

    def load_model(self, file_path: str, device: str = 'cpu'):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(file_path, map_location=torch.device(device))
        self.load_state_dict(state_dict)
        self.to(device)
        self.eval()
        print(f"Loaded model from: {file_path} on device: {device}")
class action_recognition:
    def __init__(self, device: str):
        pass
class videofolderdataset(Dataset):
    def __init__(self, folder_path: str, frames_per_video: int = 16, transform=None):
        self.folder_path = folder_path
        self.frames_per_video = frames_per_video
        self.transform = transform
        self.video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov'))]
        self.frames_per_video = frames_per_video
    def __len__(self):
        return len(self.video_files)
    def __getitem__(self, idx):
        video_path = os.path.join(self.folder_path, self.video_files[idx])
        frames = self.__extract_frames(video_path)
        if self.transform:
            frames = torch.stack([self.transform(f) for f in frames])
        return frames
    def __extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, self.frames_per_video, dtype=int)
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            else:
                frames.append(Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)))
        cap.release()
        return frames
class learnbyvideo:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
    def prepare_data(self, folder_path: str, batch_size: int = 4):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = videofolderdataset(folder_path, transform=transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    def train_on_folder(self, folder_path: str, epochs: int = 10, batch_size: int = 4):
        dataloader = self.prepare_data(folder_path, batch_size)
        for epoch in range(epochs):
            for batch in dataloader:
                batch = batch.to(self.device)
                print(f"Epoch {epoch+1}, Batch size: {batch.size(0)}")
    def train_on_url(self, video_url: str, epochs: int = 10):
        response = requests.get(video_url)
        video_path = "temp_video.mp4"
        with open(video_path, 'wb') as f:
            f.write(response.content)
        self.train_on_folder(os.path.dirname(video_path), epochs=epochs)
    def save_model(self, model ,model_name: str = "learnbyvideo.pth"):
        folder = os.path.dirname(model_name)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(model.state_dict(), model_name)
        print(f"Model saved to {model_name}")
    def load_model(self,model, path: str = "learnbyvideo.pth"):
        if not os.path.exists(path):
            print(f"No model found at {path}")
            return
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device)
        model.eval()
        print(f"Model loaded from {path}")