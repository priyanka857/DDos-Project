
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import base64

def set_png_as_page_bg(png_file):
    """
    Function to set a background image for the Streamlit app.
    """
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)

def get_base64_of_bin_file(bin_file):
    """
    Function to convert binary file data to base64.
    """
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Load the dataset
# @st.cache
def load_data():
    df = pd.read_csv('Friday-WorkingHours-Morning.pcap_ISCX.csv')
    return df

# Preprocess the data
def preprocess_data(df):
    # Encode labels
    encoder = LabelEncoder()
    df[' Label'] = encoder.fit_transform(df[' Label'])

    # Handle missing and infinite values
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)

    # Convert dataframe to integer type
    df = df.astype(int)

    # Prepare the data for training
    x = df.drop([' Label'], axis=1).values
    y = df[' Label'].values

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Convert to PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor

# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1 = None
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Adjust output classes as needed

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 128).to(x.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1 = None
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Adjust output classes as needed

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 128).to(x.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# Function to train CNN model
def train_cnn(train_loader, test_loader, num_epochs=10):
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs = inputs.unsqueeze(1)  # Add channel dimension
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        train_accuracy = correct_train / total_train

        # Validate the model
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.unsqueeze(1)  # Add channel dimension
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = correct_val / total_val

        # Append metrics to lists
        train_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return model, train_losses, train_accuracies, val_accuracies

# Function to train LSTM model
def train_lstm(train_loader, test_loader, input_size, hidden_size=128, num_layers=2, num_epochs=10):
    model = LSTMModel(input_size, hidden_size, num_layers, 2)  # Adjust output classes as needed
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs = inputs.unsqueeze(1)  # Add channel dimension
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        train_accuracy = correct_train / total_train

        # Validate the model
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.unsqueeze(1)  # Add channel dimension
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = correct_val / total_val

        # Append metrics to lists
        train_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return model, train_losses, train_accuracies, val_accuracies

def train_proposed(train_loader, test_loader, num_epochs=10):
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs = inputs.unsqueeze(1)  # Add channel dimension
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        train_accuracy = correct_train / total_train

        # Validate the model
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.unsqueeze(1)  # Add channel dimension
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = correct_val / total_val

        # Append metrics to lists
        train_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return model, train_losses, train_accuracies, val_accuracies


# Main function to run Streamlit app
def main():
    
    set_png_as_page_bg('images.jfif')
    
    
    st.title("DDoS_Attack_Detection_in_IoT_Networks")
    page = st.sidebar.selectbox("Select a page", ["Home", "CNN Model", "LSTM Model", "Proposed Model"])

    if page == "Home":
        st.header("Home Page")
        st.write("DDOS_Attack Model Training App!")
        st.write("Use the sidebar to select a page.")

    elif page == "CNN Model":
        st.header("Train CNN Model")
        st.write("Loading and preprocessing data...")
        
        # Load data
        df = load_data()
        x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor = preprocess_data(df)
        
        # Create DataLoader

        # Create DataLoader
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Train the CNN model
        st.write("Training CNN model...")
        cnn_model, cnn_train_losses, cnn_train_accuracies, cnn_val_accuracies = train_cnn(train_loader, test_loader)

        # Display training and validation metrics
        st.subheader("Training and Validation Metrics (CNN Model)")
        st.write(f"Final Training Loss: {cnn_train_losses[-1]:.4f}")
        st.write(f"Final Training Accuracy: {cnn_train_accuracies[-1]*100:.2f}%")
        st.write(f"Final Validation Accuracy: {cnn_val_accuracies[-1]*100:.2f}%")

        # Plot training and validation metrics
        st.subheader("Training Metrics Plot (CNN Model)")
        fig, ax = plt.subplots()
        ax.plot(range(1, len(cnn_train_losses) + 1), cnn_train_losses, label='Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Over Epochs (CNN Model)')
        st.pyplot(fig)

        st.subheader("Accuracy Metrics Plot (CNN Model)")
        fig, ax = plt.subplots()
        ax.plot(range(1, len(cnn_train_accuracies) + 1), cnn_train_accuracies, label='Training Accuracy')
        ax.plot(range(1, len(cnn_val_accuracies) + 1), cnn_val_accuracies, label='Validation Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training and Validation Accuracy Over Epochs (CNN Model)')
        ax.legend()
        st.pyplot(fig)

    elif page == "LSTM Model":
        st.header("Train LSTM Model")
        st.write("Loading and preprocessing data...")

        # Load data (same as above)
        df = load_data()
        x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor = preprocess_data(df)

        # Create DataLoader
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Train the LSTM model
        st.write("Training LSTM model...")
        lstm_model, lstm_train_losses, lstm_train_accuracies, lstm_val_accuracies = train_lstm(train_loader, test_loader, input_size=x_train_tensor.shape[1])

        # Display training and validation metrics
        st.subheader("Training and Validation Metrics (LSTM Model)")
        st.write(f"Final Training Loss: {lstm_train_losses[-1]:.4f}")
        st.write(f"Final Training Accuracy: {lstm_train_accuracies[-1]*100:.2f}%")
        st.write(f"Final Validation Accuracy: {lstm_val_accuracies[-1]*100:.2f}%")

        # Plot training and validation metrics
        st.subheader("Training Metrics Plot (LSTM Model)")
        fig, ax = plt.subplots()
        ax.plot(range(1, len(lstm_train_losses) + 1), lstm_train_losses, label='Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Over Epochs (LSTM Model)')
        st.pyplot(fig)

        st.subheader("Accuracy Metrics Plot (LSTM Model)")
        fig, ax = plt.subplots()
        ax.plot(range(1, len(lstm_train_accuracies) + 1), lstm_train_accuracies, label='Training Accuracy')
        ax.plot(range(1, len(lstm_val_accuracies) + 1), lstm_val_accuracies, label='Validation Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training and Validation Accuracy Over Epochs (LSTM Model)')
        ax.legend()
        st.pyplot(fig)
        
    elif page == "Proposed Model":
        st.header("Train Proposed Model")
        st.write("Loading and preprocessing data...")
        
        # Load data
        df = load_data()
        x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor = preprocess_data(df)
        
        # Create DataLoader

        # Create DataLoader
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Train the CNN model
        st.write("Training Proposed model...")
        cnn_model, cnn_train_losses, cnn_train_accuracies, cnn_val_accuracies = train_cnn(train_loader, test_loader)

        # Display training and validation metrics
        st.subheader("Training and Validation Metrics (Proposed model)")
        st.write(f"Final Training Loss: {cnn_train_losses[-1]:.4f}")
        st.write(f"Final Training Accuracy: {cnn_train_accuracies[-1]*100:.2f}%")
        st.write(f"Final Validation Accuracy: {cnn_val_accuracies[-1]*100:.2f}%")

        # Plot training and validation metrics
        st.subheader("Training Metrics Plot (Proposed model)")
        fig, ax = plt.subplots()
        ax.plot(range(1, len(cnn_train_losses) + 1), cnn_train_losses, label='Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Over Epochs (Proposed model)')
        st.pyplot(fig)

        st.subheader("Accuracy Metrics Plot (Proposed model)")
        fig, ax = plt.subplots()
        ax.plot(range(1, len(cnn_train_accuracies) + 1), cnn_train_accuracies, label='Training Accuracy')
        ax.plot(range(1, len(cnn_val_accuracies) + 1), cnn_val_accuracies, label='Validation Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training and Validation Accuracy Over Epochs (Proposed model)')
        ax.legend()
        st.pyplot(fig)


if __name__ == "__main__":
    main()