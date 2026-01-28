import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.utils.data import DataLoader, TensorDataset, random_split
import math
import pickle
import random
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import os

# Set random seed for reproducibility
random.seed(66)
np.random.seed(66)
torch.manual_seed(66)
torch.cuda.manual_seed_all(66)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# Global hyperparameters
INPUT_CHANNELS = 5
CONV1_OUT_CHANNELS = 65
CONV1_KERNEL_SIZE = 13
CONV2_OUT_CHANNELS = 130
CONV2_KERNEL_SIZE = 11
CONV3_OUT_CHANNELS = 260
CONV3_KERNEL_SIZE = 9
CONV3_DILATION = 2
CONV4_OUT_CHANNELS = 520
CONV4_KERNEL_SIZE = 7
CONV4_DILATION = 2
SE_REDUCTION = 16
ATTENTION_EMBED_DIM = 520
ATTENTION_NUM_HEADS = 10
ATTENTION_DROPOUT = 0.1
LSTM_HIDDEN_SIZE = 255
LSTM_NUM_LAYERS = 1
FC1_OUT_FEATURES = 255
DROPOUT1 = 0.4
FC2_OUT_FEATURES = 65
DROPOUT2 = 0.3
BATCH_SIZE = 128
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
NUM_WORKERS_TRAIN = 6
NUM_WORKERS_VAL_TEST = 2
INITIAL_LR = 3.1336e-5
INITIAL_WD = 10.7e-3
HUBER_DELTA = 1.0
THRESHOLD_RATIO = 1.3
WD_INCREASE_FACTOR = 1.005
MAX_WD = 1e-1
NUM_EPOCHS = 90
MAX_LR = 4e-4
PCT_START = 0.16
FINAL_DIV_FACTOR = 15
SEQ_LEN = 500
LAMBDA_QI_VALUES = [0.001]  # ,100 ,200, 300, 400, 500

def worker_init_fn(worker_id):
    np.random.seed(66 + worker_id)

def get_same_padding(kernel_size, dilation=1):
    return (dilation * (kernel_size - 1)) // 2

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L]

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=SE_REDUCTION):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(channel // reduction, 1), bias=False),
            nn.GELU(),
            nn.Linear(max(channel // reduction, 1), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class ResNet_BML_MultiChannel(nn.Module):
    def __init__(self, seq_len=SEQ_LEN):
        super(ResNet_BML_MultiChannel, self).__init__()
        self.seq_len = seq_len

        # Convolutional layers
        self.conv1 = nn.Conv1d(INPUT_CHANNELS, CONV1_OUT_CHANNELS, kernel_size=CONV1_KERNEL_SIZE, padding=get_same_padding(CONV1_KERNEL_SIZE))
        self.conv2 = nn.Conv1d(CONV1_OUT_CHANNELS, CONV2_OUT_CHANNELS, kernel_size=CONV2_KERNEL_SIZE, padding=get_same_padding(CONV2_KERNEL_SIZE))
        self.conv3 = nn.Conv1d(CONV2_OUT_CHANNELS, CONV3_OUT_CHANNELS, kernel_size=CONV3_KERNEL_SIZE, dilation=CONV3_DILATION, padding=get_same_padding(CONV3_KERNEL_SIZE, CONV3_DILATION))
        self.conv4 = nn.Conv1d(CONV3_OUT_CHANNELS, CONV4_OUT_CHANNELS, kernel_size=CONV4_KERNEL_SIZE, dilation=CONV4_DILATION, padding=get_same_padding(CONV4_KERNEL_SIZE, CONV4_DILATION))

        # Residual connection adjustment layers
        self.adjust1 = nn.Conv1d(INPUT_CHANNELS, CONV1_OUT_CHANNELS, kernel_size=1)
        self.adjust2 = nn.Conv1d(CONV1_OUT_CHANNELS, CONV2_OUT_CHANNELS, kernel_size=1)
        self.adjust3 = nn.Conv1d(CONV2_OUT_CHANNELS, CONV3_OUT_CHANNELS, kernel_size=1)
        self.adjust4 = nn.Conv1d(CONV3_OUT_CHANNELS, CONV4_OUT_CHANNELS, kernel_size=1)

        # Squeeze-and-Excitation modules
        self.se1 = SEBlock(CONV1_OUT_CHANNELS, reduction=SE_REDUCTION)
        self.se2 = SEBlock(CONV2_OUT_CHANNELS, reduction=SE_REDUCTION)
        self.se3 = SEBlock(CONV3_OUT_CHANNELS, reduction=SE_REDUCTION)
        self.se4 = SEBlock(CONV4_OUT_CHANNELS, reduction=SE_REDUCTION)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model=ATTENTION_EMBED_DIM, max_len=seq_len)

        in_feat = CONV4_OUT_CHANNELS
        if in_feat != ATTENTION_EMBED_DIM:
            self.input_proj = nn.Linear(in_feat, ATTENTION_EMBED_DIM)
        else:
            self.input_proj = nn.Identity()

        # Pre-normalized multi-head attention
        self.norm = nn.LayerNorm(ATTENTION_EMBED_DIM)
        self.attn = nn.MultiheadAttention(embed_dim=ATTENTION_EMBED_DIM, num_heads=ATTENTION_NUM_HEADS, dropout=ATTENTION_DROPOUT, batch_first=True)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_size=ATTENTION_EMBED_DIM, hidden_size=LSTM_HIDDEN_SIZE,
                            num_layers=LSTM_NUM_LAYERS, batch_first=True, bidirectional=True)

        # Classification head
        self.fc1 = nn.Linear(LSTM_HIDDEN_SIZE * 2, FC1_OUT_FEATURES)
        self.dropout1 = nn.Dropout(DROPOUT1)
        self.fc2 = nn.Linear(FC1_OUT_FEATURES, FC2_OUT_FEATURES)
        self.dropout2 = nn.Dropout(DROPOUT2)
        self.fc3 = nn.Linear(FC2_OUT_FEATURES, 1)
        self.initialize_weights()

    def forward(self, x):
        residual = x
        x = nn.GELU()(self.conv1(x))
        x = x + self.adjust1(residual)
        x = self.se1(x)

        residual = x
        x = nn.GELU()(self.conv2(x))
        x = x + self.adjust2(residual)
        x = self.se2(x)

        residual = x
        x = nn.GELU()(self.conv3(x))
        x = x + self.adjust3(residual)
        x = self.se3(x)

        residual = x
        x = nn.GELU()(self.conv4(x))
        x = x + self.adjust4(residual)
        x = self.se4(x)

        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        x = self.pos_encoding(x)

        x_norm = self.norm(x)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + x_attn

        lstm_out, _ = self.lstm(x)
        x_pool = torch.mean(lstm_out, dim=1)

        x_fc = nn.GELU()(self.fc1(x_pool))
        x_fc = self.dropout1(x_fc)
        x_fc = nn.GELU()(self.fc2(x_fc))
        x_fc = self.dropout2(x_fc)
        out = self.fc3(x_fc)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                try:
                    nn.init.xavier_normal_(m.weight)
                except Exception as e:
                    print(f"Warning: Xavier init failed for {type(m).__name__}: {e}")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.orthogonal_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

def calculate_qi(temperature_data, h_predicted):
    try:
        h = int(np.round(h_predicted))
        h_extended = int(np.round(1.5 * h))
        max_depth = len(temperature_data)
        if h <= 0 or h_extended > max_depth or h_extended <= h:
            return np.nan
        var_x = np.var(temperature_data[:h])
        var_y = np.var(temperature_data[:h_extended])
        if var_y <= 0:
            return np.nan
        qi = 1 - var_x / var_y
        return qi
    except Exception as e:
        print(f"QI calculation error: {e}")
        return np.nan

def calculate_qi_tensor(temp_profiles, h_pred, k=10.0, eps=1e-8):
    B, L = temp_profiles.shape
    device = temp_profiles.device
    h_pred = h_pred.squeeze(-1)
    # Clamp h_pred to valid range to reduce NaN
    h_pred = torch.clamp(h_pred, min=1.0, max=float(L) / 1.5)

    idx = torch.arange(L, device=device).float().view(1, -1)

    mask_x = torch.sigmoid(k * (h_pred.unsqueeze(1) - idx))
    mask_y = torch.sigmoid(k * ((1.5 * h_pred).unsqueeze(1) - idx))

    def weighted_var(x, mask):
        sum_w = mask.sum(dim=1, keepdim=True) + eps
        mean = (x * mask).sum(dim=1, keepdim=True) / sum_w
        mean_sq = (x * x * mask).sum(dim=1, keepdim=True) / sum_w
        return mean_sq - mean * mean

    var_x = weighted_var(temp_profiles, mask_x)
    var_y = weighted_var(temp_profiles, mask_y)

    qi = 1.0 - var_x / (var_y + eps)
    return qi

def save_plot_as_pkl(fig, filename='plot.pkl'):
    data = {
        "lines": [line.get_ydata() for ax in fig.axes for line in ax.get_lines()],
        "titles": [ax.get_title() for ax in fig.axes],
        "labels": [(ax.get_xlabel(), ax.get_ylabel()) for ax in fig.axes]
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def safe_metrics(sorted_targets, sorted_preds):
    targets = np.array(sorted_targets).astype(float)
    preds = np.array(sorted_preds).astype(float)
    # Filter NaN and Inf for robustness
    valid_mask = ~np.isnan(targets) & ~np.isnan(preds) & ~np.isinf(targets) & ~np.isinf(preds)
    targets = targets[valid_mask]
    preds = preds[valid_mask]
    n = targets.size
    if n == 0:
        return 0.0, 0.0, float('inf'), float('inf')
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    rmse = np.sqrt(np.mean((targets - preds) ** 2))
    mae = np.mean(np.abs(targets - preds))
    denom = (np.std(targets) * np.std(preds) * n)
    if denom == 0:
        pearson = 0.0
    else:
        pearson = np.corrcoef(targets, preds)[0,1]
    return r2, pearson, rmse, mae

def train_and_evaluate(lambda_qi, train_loader, val_loader, test_loader, device, temp_data):
    random.seed(66)
    np.random.seed(66)
    torch.manual_seed(66)
    torch.cuda.manual_seed_all(66)
    model = ResNet_BML_MultiChannel(seq_len=SEQ_LEN).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=INITIAL_WD)
    criterion = nn.HuberLoss(delta=HUBER_DELTA)
    total_steps = max(1, NUM_EPOCHS * max(1, len(train_loader)))
    scheduler = OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        total_steps=total_steps,
        pct_start=PCT_START,
        final_div_factor=FINAL_DIV_FACTOR,
        anneal_strategy='cos',
        cycle_momentum=False
    )

    best_val_loss = float('inf')
    best_model_weights = None
    best_epoch = 0
    train_losses, val_losses = [], []

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        batch_count = 0

        for inputs, targets, qi_trues, _, _ in train_loader:
            inputs, targets, qi_trues = inputs.to(device), targets.to(device), qi_trues.to(device)
            optimizer.zero_grad()
            try:
                outputs = model(inputs)
            except Exception as e:
                print(f"Error during forward pass in training (lambda_qi={lambda_qi}):", e)
                raise

            loss_ml = criterion(outputs, targets)
            temp_profiles = inputs[:, 0, :]
            qi_pred = calculate_qi_tensor(temp_profiles, outputs)
            # Fix: Filter both qi_trues and qi_pred for NaN in training
            valid_mask = ~torch.isnan(qi_trues) & ~torch.isnan(qi_pred)
            if valid_mask.sum() > 0:
                loss_qi = F.huber_loss(qi_pred[valid_mask], qi_trues[valid_mask], delta=HUBER_DELTA, reduction='mean')
            else:
                loss_qi = torch.tensor(0.0, device=device)

            loss = loss_ml + lambda_qi * loss_qi
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            batch_count += 1

        if batch_count > 0:
            train_loss /= batch_count
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for inputs, targets, qi_trues, _, _ in val_loader:
                inputs, targets, qi_trues = inputs.to(device), targets.to(device), qi_trues.to(device)
                outputs = model(inputs)
                loss_ml = criterion(outputs, targets)
                temp_profiles = inputs[:, 0, :]
                qi_pred = calculate_qi_tensor(temp_profiles, outputs)
                valid_mask = ~torch.isnan(qi_trues) & ~torch.isnan(qi_pred)
                if valid_mask.sum() > 0:
                    loss_qi = F.huber_loss(qi_pred[valid_mask], qi_trues[valid_mask], delta=HUBER_DELTA, reduction='mean')
                else:
                    loss_qi = torch.tensor(0.0, device=device)
                loss = loss_ml + lambda_qi * loss_qi
                val_loss += loss.item()
                val_count += 1
        if val_count > 0:
            val_loss /= val_count
        val_losses.append(val_loss)

        if val_loss < best_val_loss and not math.isnan(val_loss):
            best_val_loss = val_loss
            best_model_weights = model.state_dict()
            best_epoch = epoch + 1

        if not (math.isnan(train_loss) or math.isnan(val_loss)) and train_loss > 0:
            ratio = val_loss / train_loss
            if ratio > THRESHOLD_RATIO:
                for param_group in optimizer.param_groups:
                    new_wd = param_group['weight_decay'] * WD_INCREASE_FACTOR
                    param_group['weight_decay'] = min(new_wd, MAX_WD)

        print(f"Lambda_QI {lambda_qi} | Epoch {epoch + 1}/{NUM_EPOCHS} — Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # Evaluate on test set
    model.load_state_dict(best_model_weights)
    model.eval()
    pred_list, target_list, idx_list, mean_temps_list = [], [], [], []
    with torch.no_grad():
        for inputs, target, qi_trues, idx, mean_temp in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            output = model(inputs).cpu().numpy()
            pred_list.extend(output.flatten())
            target_list.extend(target.numpy().flatten())
            idx_list.extend(idx.numpy().flatten())
            mean_temps_list.extend(mean_temp.numpy().flatten())

    sorted_indices = np.argsort(mean_temps_list)
    sorted_test_idxs = np.array(idx_list)[sorted_indices]
    sorted_targets = np.array(target_list)[sorted_indices]
    sorted_preds = np.array(pred_list)[sorted_indices]

    try:
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        from scipy.stats import pearsonr
        r2 = r2_score(sorted_targets, sorted_preds)
        pearson, _ = pearsonr(sorted_targets, sorted_preds)
        rmse = np.sqrt(mean_squared_error(sorted_targets, sorted_preds))
        mae = mean_absolute_error(sorted_targets, sorted_preds)
    except Exception as e:
        print(f"Error computing sklearn metrics for lambda_qi={lambda_qi}: {e}")
        r2, pearson, rmse, mae = safe_metrics(sorted_targets, sorted_preds)

    qi_pred_list = []
    for i in range(len(sorted_test_idxs)):
        temp_profile = temp_data[sorted_test_idxs[i], :].flatten()
        qi_pred_v = calculate_qi(temp_profile, sorted_preds[i])
        qi_pred_list.append(qi_pred_v)
    avg_qi_pred = np.nanmean(qi_pred_list)

    return {
        'lambda_qi': lambda_qi,
        'val_loss': best_val_loss,
        'best_model_state_dict': best_model_weights,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'r2': r2,
        'pearson': pearson,
        'rmse': rmse,
        'mae': mae,
        'avg_qi_pred': avg_qi_pred,
        'best_epoch': best_epoch,
        'sorted_targets': sorted_targets,
        'sorted_preds': sorted_preds,
        'sorted_idxs': sorted_test_idxs
    }

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    TEMP_PATH = r"D:\WOCE data_\1temp\temp.mat"
    ML_PATH = r"D:\WOCE data_\1temp\ML.mat"
    OUTPUT_DIR = r"D:\WOCE data_\1temp\basic+qiloss"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === 加载数据 ===
    temp_data = sio.loadmat(TEMP_PATH)['temp'][:, :SEQ_LEN]
    ml_data = sio.loadmat(ML_PATH)['ML'].squeeze()

    print(f"Temp data shape: {temp_data.shape}")
    print(f"ML data shape: {ml_data.shape}")

    # 计算平均温度用于排序
    mean_temps = np.mean(temp_data, axis=1)

    # 计算真实 QI
    qi_true_list = []
    for i in range(len(ml_data)):
        qi_true_list.append(calculate_qi(temp_data[i, :], ml_data[i]))
    qi_true = np.array(qi_true_list, dtype=np.float32)

    # 计算附加通道
    grad_data = np.gradient(temp_data, axis=1)
    grad2_data = np.gradient(grad_data, axis=1)
    var_data = np.array([np.var(temp_data[:, :i + 1], axis=1) for i in range(temp_data.shape[1])]).T
    var_grad_data = np.gradient(var_data, axis=1)
    input_data = np.stack([temp_data, grad_data, grad2_data, var_data, var_grad_data], axis=1)

    # 标准化（全局 fit 所有数据，或每折 fit 训练集？推荐：每折 fit 训练集）
    input_tensor_raw = input_data.copy()
    ml_tensor = torch.tensor(ml_data, dtype=torch.float32).view(-1, 1)
    qi_tensor = torch.tensor(qi_true, dtype=torch.float32).view(-1, 1)
    indices_tensor = torch.arange(len(input_data))
    mean_temps_tensor = torch.tensor(mean_temps, dtype=torch.float32)

    # === K 折交叉验证设置 ===
    K = 5
    kf = KFold(n_splits=K, shuffle=True, random_state=66)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 存储每折结果
    fold_results = []
    all_sorted_preds = []
    all_sorted_targets = []
    all_sorted_idxs = []
    all_mean_temps = []
    all_fold_ids = []  # 新增：记录每个测试样本来自哪个fold

    print(f"\n{'='*60}")
    print(f" 开始 {K}-Fold 交叉验证")
    print(f"{'='*60}")

    for fold, (train_val_idx, test_idx) in enumerate(kf.split(input_data)):
        print(f"\n{'-'*50}")
        print(f" Fold {fold + 1}/{K}")
        print(f"{'-'*50}")

        # 进一步划分 train/val
        train_idx, val_idx = [], []
        tv_idx = np.array(train_val_idx)
        np.random.seed(66 + fold)
        perm = np.random.permutation(len(tv_idx))
        train_size = int(0.889 * len(tv_idx))  # 80/90 of train_val
        train_idx = tv_idx[perm[:train_size]]
        val_idx = tv_idx[perm[train_size:]]

        # === 每折标准化（关键！避免数据泄露）===
        scaler = StandardScaler()
        input_train = input_data[train_idx]
        input_train_reshaped = input_train.transpose(0, 2, 1).reshape(-1, INPUT_CHANNELS)
        scaler.fit(input_train_reshaped)

        # 标准化所有数据
        input_all_reshaped = input_data.transpose(0, 2, 1).reshape(-1, INPUT_CHANNELS)
        input_standardized = scaler.transform(input_all_reshaped)
        input_standardized = input_standardized.reshape(len(input_data), SEQ_LEN, INPUT_CHANNELS).transpose(0, 2, 1)
        input_tensor = torch.tensor(input_standardized, dtype=torch.float32)

        # 更新数据集
        dataset = TensorDataset(input_tensor, ml_tensor, qi_tensor, indices_tensor, mean_temps_tensor)
        train_set = torch.utils.data.Subset(dataset, train_idx)
        val_set = torch.utils.data.Subset(dataset, val_idx)
        test_set = torch.utils.data.Subset(dataset, test_idx)

        # DataLoaders
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS_TRAIN, pin_memory=True, worker_init_fn=worker_init_fn)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS_VAL_TEST, pin_memory=True, worker_init_fn=worker_init_fn)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS_VAL_TEST, pin_memory=True, worker_init_fn=worker_init_fn)

        # === 训练与评估（支持多个 lambda_qi）===
        best_lambda_qi = None
        best_val_loss = float('inf')
        best_result = None
        fold_lambda_results = []

        for lambda_qi in LAMBDA_QI_VALUES:
            print(f"\n  → Training with lambda_qi = {lambda_qi}")
            result = train_and_evaluate(
                lambda_qi=lambda_qi,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                temp_data=temp_data
            )
            fold_lambda_results.append(result)

            if result['val_loss'] < best_val_loss and not math.isnan(result['val_loss']):
                best_val_loss = result['val_loss']
                best_lambda_qi = lambda_qi
                best_result = result

        # 保存本折最佳模型权重（文件名包含fold和lambda）
        model_save_path = os.path.join(OUTPUT_DIR, f"fold_{fold + 1}_best_model_lambda_{best_lambda_qi}.pth")
        torch.save(best_result['best_model_state_dict'], model_save_path)
        print(f"Saved best model for fold {fold + 1} (lambda_qi={best_lambda_qi}) to {model_save_path}")

        # 保存每个fold的独立预测结果（包含fold_id）
        fold_result_path = os.path.join(OUTPUT_DIR, f"fold_{fold + 1}_predictions.mat")
        fold_results_mat = np.column_stack((
            best_result['sorted_idxs'] + 1,           # 站点索引 (从1开始)
            best_result['sorted_targets'],            # 真实值
            best_result['sorted_preds'],              # 预测值
            np.full(len(best_result['sorted_idxs']), fold + 1)  # 该fold的ID
        ))
        sio.savemat(fold_result_path, {'Fold_Results': fold_results_mat})
        print(f"Saved fold {fold + 1} predictions to {fold_result_path}")

        # 保存本折最佳结果（不包含模型权重）
        del best_result['best_model_state_dict']
        fold_results.append(best_result)

        # 收集测试集预测（用于最终汇总）
        all_sorted_preds.extend(best_result['sorted_preds'])
        all_sorted_targets.extend(best_result['sorted_targets'])
        all_sorted_idxs.extend(best_result['sorted_idxs'])
        all_mean_temps.extend(mean_temps[best_result['sorted_idxs']])
        all_fold_ids.extend([fold + 1] * len(best_result['sorted_preds']))  # 记录该样本来自哪个fold

        print(f" Fold {fold+1} Best lambda_qi: {best_lambda_qi} | Val Loss: {best_val_loss:.6f}")
        print(f" R²: {best_result['r2']:.4f} | RMSE: {best_result['rmse']:.4f} | MAE: {best_result['mae']:.4f}")

    # === K 折汇总统计 ===
    print(f"\n{'='*60}")
    print(f" {K}-FOLD CROSS VALIDATION SUMMARY")
    print(f"{'='*60}")
    print("| Fold | Lambda_QI | R²    | Pearson | RMSE  | MAE   | Avg QI | Val Loss |")
    print("|------|-----------|-------|---------|-------|-------|--------|----------|")

    r2s = [r['r2'] for r in fold_results]
    pearsons = [r['pearson'] for r in fold_results]
    rmses = [r['rmse'] for r in fold_results]
    maes = [r['mae'] for r in fold_results]
    avg_qis = [r['avg_qi_pred'] for r in fold_results]
    val_losses = [r['val_loss'] for r in fold_results]

    for i, r in enumerate(fold_results):
        print(f"| {i+1:<4} | {r['lambda_qi']:<9} | {r['r2']:.4f} | {r['pearson']:.4f} | "
              f"{r['rmse']:.4f} | {r['mae']:.4f} | {r['avg_qi_pred']:.4f} | {r['val_loss']:.6f} |")

    print(f"{'-'*60}")
    print(f"| MEAN | -         | {np.mean(r2s):.4f} | {np.mean(pearsons):.4f} | {np.mean(rmses):.4f} | "
          f"{np.mean(maes):.4f} | {np.mean(avg_qis):.4f} | {np.mean(val_losses):.6f} |")
    print(f"| STD  | -         | ±{np.std(r2s):.4f} | ±{np.std(pearsons):.4f} | ±{np.std(rmses):.4f} | "
          f"±{np.std(maes):.4f} | ±{np.std(avg_qis):.4f} | ±{np.std(val_losses):.6f} |")
    print(f"{'='*60}")

    # === 最终绘图与结果保存（包含fold信息）===
    sorted_order = np.argsort(all_mean_temps)
    final_sorted_preds = np.array(all_sorted_preds)[sorted_order]
    final_sorted_targets = np.array(all_sorted_targets)[sorted_order]
    final_sorted_idxs = np.array(all_sorted_idxs)[sorted_order]
    final_sorted_fold_ids = np.array(all_fold_ids)[sorted_order]  # 对应的fold ID
    final_test_temp_data = temp_data[final_sorted_idxs, :]

    # 保存最终汇总结果（包含哪个fold的模型预测了该站点）
    result_path = os.path.join(OUTPUT_DIR, f"result_5channel_kfold{K}_final.mat")
    results_mat = np.column_stack((
        final_sorted_idxs + 1,          # 第1列：站点索引 (从1开始)
        final_sorted_targets,           # 第2列：真实ML深度
        final_sorted_preds,             # 第3列：预测ML深度
        final_sorted_fold_ids           # 第4列：使用哪个fold的模型（1~5）
    ))
    sio.savemat(result_path, {'Results': results_mat})
    print(f"\nFinal results (with fold model info) saved to {result_path}")
    print("MAT文件列说明：")
    print("  列1: 站点索引 (原数据顺序，从1开始)")
    print("  列2: 真实ML深度")
    print("  列3: 模型预测ML深度")
    print("  列4: 该站点预测所用的模型对应Fold ID (1~5，对应保存的fold_X_best_model_*.pth)")

    # === 最终指标 ===
    r2_final = r2_score(final_sorted_targets, final_sorted_preds)
    pearson_final, _ = pearsonr(final_sorted_targets, final_sorted_preds)
    rmse_final = np.sqrt(mean_squared_error(final_sorted_targets, final_sorted_preds))
    mae_final = mean_absolute_error(final_sorted_targets, final_sorted_preds)

    print("\n### FINAL AGGREGATED TEST PERFORMANCE ###")
    print(f"R²: {r2_final:.4f}")
    print(f"Pearson: {pearson_final:.4f}")
    print(f"RMSE: {rmse_final:.4f}")
    print(f"MAE: {mae_final:.4f}")

    # === 绘图（保持不变）===
    fig, axes = plt.subplots(4, 1, figsize=(14, 20))

    # 1. K-fold Val Loss 对比
    for i, r in enumerate(fold_results):
        epochs = np.arange(1, len(r['val_losses']) + 1)
        axes[0].plot(epochs, r['val_losses'], label=f'Fold {i+1} (λ={r["lambda_qi"]})')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation Loss')
    axes[0].set_title(f'{K}-Fold Validation Loss Curves')
    axes[0].legend()
    axes[0].grid(True)

    # 2. 最终预测 vs 真实
    x_coords = np.arange(len(final_sorted_targets))
    cax = axes[1].imshow(final_test_temp_data.T, aspect='auto', cmap=cm.jet, origin='upper')
    fig.colorbar(cax, ax=axes[1], label='Temperature')
    axes[1].plot(x_coords, final_sorted_targets, 'k-', label='ML (True)')
    axes[1].plot(x_coords, final_sorted_preds, 'gray', linestyle='--', label='Predicted')
    axes[1].set_xlabel('Test Samples (Sorted by Mean Temp)')
    axes[1].set_ylabel('Depth')
    axes[1].set_title('Temperature Field with Predictions')
    axes[1].legend()
    axes[1].set_ylim(0, SEQ_LEN - 1)

    # 3. 散点图
    axes[2].scatter(final_sorted_targets, final_sorted_preds, alpha=0.5, s=10)
    axes[2].plot([final_sorted_targets.min(), final_sorted_targets.max()],
                 [final_sorted_targets.min(), final_sorted_targets.max()], 'r--')
    axes[2].set_xlabel('True ML Depth')
    axes[2].set_ylabel('Predicted ML Depth')
    axes[2].set_title(f'Prediction vs True (R²={r2_final:.3f})')
    axes[2].grid(True)

    # 4. 单站温度剖面对比（交互）
    def plot_profile(ax, idx):
        orig_idx = final_sorted_idxs[idx]
        ax.clear()
        ax.plot(temp_data[orig_idx], np.arange(SEQ_LEN), 'b-', label='Temperature')
        ax.axhline(final_sorted_targets[idx], color='k', label='ML')
        ax.axhline(final_sorted_preds[idx], color='gray', linestyle='--', label='Pred')
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Depth')
        ax.set_title(f'Station {orig_idx+1} (Predicted by Fold {final_sorted_fold_ids[idx]})')
        ax.invert_yaxis()
        ax.legend()
        ax.grid(True)

    if len(final_sorted_idxs) > 0:
        plot_profile(axes[3], 0)

    def on_click(event):
        if event.inaxes == axes[1]:
            if event.xdata is not None:
                idx = int(round(event.xdata))
                idx = max(0, min(idx, len(final_sorted_idxs) - 1))
                plot_profile(axes[3], idx)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.tight_layout()

    # 保存图像
    png_path = os.path.join(OUTPUT_DIR, f"plot_kfold{K}_final.png")
    pkl_path = os.path.join(OUTPUT_DIR, f"plot_kfold{K}_final.pkl")
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    save_plot_as_pkl(fig, pkl_path)
    print(f"Final plot saved to {png_path} and {pkl_path}")

    plt.show()

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()