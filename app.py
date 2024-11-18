import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np
import pandas as pd
from datetime import datetime
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ModernUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CyberGuard ML Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f2f5")
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('Modern.TButton',
                           padding=10,
                           background='#0066cc',
                           foreground='white')
        self.style.configure('Alert.TButton',
                           padding=10,
                           background='#dc3545',
                           foreground='white')
        
        # Initialize model
        self.model, self.accuracy = self.train_model()
        
        self.create_dashboard()
        self.create_monitoring_panel()
        self.create_alert_panel()
        self.create_settings_panel()
        
        # Start monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.simulate_traffic)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        # Initialize data for visualization
        self.traffic_data = {'timestamp': [], 'bytes': [], 'duration': [], 'packet_size': []}

    def create_dashboard(self):
        # Main dashboard frame
        dashboard = ttk.Frame(self.root, padding="20")
        dashboard.pack(fill=tk.BOTH, expand=True)

        # Header
        header = ttk.Frame(dashboard)
        header.pack(fill=tk.X, pady=(0, 20))
        
        title = ttk.Label(header, 
                         text="CyberGuard ML Detection System",
                         font=("Helvetica", 24, "bold"))
        title.pack(side=tk.LEFT)
        
        self.accuracy_label = ttk.Label(header,
                                       text=f"Model Accuracy: {self.accuracy:.2f}%",
                                       font=("Helvetica", 12))
        self.accuracy_label.pack(side=tk.RIGHT, pady=10)

        # Create tabs
        self.tab_control = ttk.Notebook(dashboard)
        
        self.dashboard_tab = ttk.Frame(self.tab_control)
        self.analysis_tab = ttk.Frame(self.tab_control)
        self.monitoring_tab = ttk.Frame(self.tab_control)
        self.settings_tab = ttk.Frame(self.tab_control)
        
        self.tab_control.add(self.dashboard_tab, text='Dashboard')
        self.tab_control.add(self.analysis_tab, text='Analysis')
        self.tab_control.add(self.monitoring_tab, text='Real-time Monitoring')
        self.tab_control.add(self.settings_tab, text='Settings')
        
        self.tab_control.pack(expand=True, fill=tk.BOTH)

        # Create dashboard content
        self.create_dashboard_content()

    def create_dashboard_content(self):
        # Summary statistics
        summary_frame = ttk.LabelFrame(self.dashboard_tab, text="Summary", padding="20")
        summary_frame.pack(fill=tk.X, padx=20, pady=20)

        self.total_traffic_label = ttk.Label(summary_frame, text="Total Traffic: 0", font=("Helvetica", 12))
        self.total_traffic_label.pack(side=tk.LEFT, padx=20)

        self.alerts_label = ttk.Label(summary_frame, text="Total Alerts: 0", font=("Helvetica", 12))
        self.alerts_label.pack(side=tk.LEFT, padx=20)

        # Traffic visualization
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.dashboard_tab)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    def create_monitoring_panel(self):
        # Real-time monitoring panel
        monitor_frame = ttk.LabelFrame(self.monitoring_tab,
                                     text="Network Traffic Monitor",
                                     padding="20")
        monitor_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Traffic visualization
        self.traffic_text = tk.Text(monitor_frame,
                                  height=10,
                                  wrap=tk.WORD,
                                  font=("Courier", 10))
        self.traffic_text.pack(fill=tk.BOTH, expand=True)
        
        # Manual check frame
        manual_frame = ttk.LabelFrame(self.analysis_tab,
                                    text="Manual Traffic Analysis",
                                    padding="20")
        manual_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Input fields
        input_frame = ttk.Frame(manual_frame)
        input_frame.pack(fill=tk.X, pady=10)

        ttk.Label(input_frame, text="Bytes:").pack(side=tk.LEFT)
        self.bytes_entry = ttk.Entry(input_frame, width=20)
        self.bytes_entry.pack(side=tk.LEFT, padx=5)

        ttk.Label(input_frame, text="Duration:").pack(side=tk.LEFT, padx=(20, 0))
        self.duration_entry = ttk.Entry(input_frame, width=20)
        self.duration_entry.pack(side=tk.LEFT, padx=5)

        ttk.Label(input_frame, text="Packet Size:").pack(side=tk.LEFT, padx=(20, 0))
        self.packet_entry = ttk.Entry(input_frame, width=20)
        self.packet_entry.pack(side=tk.LEFT, padx=5)

        # Analyze button
        analyze_btn = ttk.Button(manual_frame,
                               text="Analyze Traffic",
                               style='Modern.TButton',
                               command=self.analyze_traffic)
        analyze_btn.pack(pady=20)

    def create_alert_panel(self):
        # Alert panel
        self.alert_frame = ttk.LabelFrame(self.analysis_tab,
                                        text="Alert History",
                                        padding="20")
        self.alert_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Alert list
        self.alert_text = tk.Text(self.alert_frame,
                                height=8,
                                wrap=tk.WORD,
                                font=("Helvetica", 10))
        self.alert_text.pack(fill=tk.BOTH, expand=True)

        # Export alerts button
        export_btn = ttk.Button(self.alert_frame,
                              text="Export Alerts",
                              style='Modern.TButton',
                              command=self.export_alerts)
        export_btn.pack(pady=10)

    def create_settings_panel(self):
        settings_frame = ttk.LabelFrame(self.settings_tab,
                                      text="Settings",
                                      padding="20")
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Threshold setting
        threshold_frame = ttk.Frame(settings_frame)
        threshold_frame.pack(fill=tk.X, pady=10)

        ttk.Label(threshold_frame, text="Alert Threshold:").pack(side=tk.LEFT)
        self.threshold_entry = ttk.Entry(threshold_frame, width=10)
        self.threshold_entry.insert(0, "0.5")  # Default threshold
        self.threshold_entry.pack(side=tk.LEFT, padx=5)

        # Update threshold button
        update_btn = ttk.Button(threshold_frame,
                              text="Update Threshold",
                              style='Modern.TButton',
                              command=self.update_threshold)
        update_btn.pack(side=tk.LEFT, padx=10)

        # Retrain model button
        retrain_btn = ttk.Button(settings_frame,
                               text="Retrain Model",
                               style='Modern.TButton',
                               command=self.retrain_model)
        retrain_btn.pack(pady=20)

    def train_model(self):
        X, y = make_classification(n_samples=1000,
                                 n_features=3,
                                 n_informative=2,
                                 n_redundant=0,
                                 random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                           test_size=0.2,
                                                           random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test) * 100
        return model, accuracy

    def analyze_traffic(self):
        try:
            features = [
                float(self.bytes_entry.get()),
                float(self.duration_entry.get()),
                float(self.packet_entry.get())
            ]
            
            prediction = self.model.predict([features])[0]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if prediction == 1:
                alert_msg = f"⚠️ [{timestamp}] ALERT: Malicious Activity Detected!\n"
                self.alert_text.insert(tk.END, alert_msg)
                self.alert_text.see(tk.END)
                messagebox.showwarning("Security Alert",
                                     "Malicious activity detected! Taking protective action...")
            else:
                self.traffic_text.insert(tk.END,
                                       f"✔️ [{timestamp}] Benign traffic detected\n")
                self.traffic_text.see(tk.END)
                
        except ValueError:
            messagebox.showerror("Error",
                               "Please enter valid numeric values for all fields")

    def simulate_traffic(self):
        """Simulate real-time traffic for monitoring"""
        total_traffic = 0
        total_alerts = 0
        while self.monitoring:
            # Generate random traffic data
            bytes_val = np.random.randint(100, 10000)
            duration = np.random.uniform(0.1, 5.0)
            packet_size = np.random.randint(50, 1500)
            features = [bytes_val, duration, packet_size]
            
            prediction = self.model.predict([features])[0]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            total_traffic += 1
            if prediction == 1:
                alert_msg = f"⚠️ [{timestamp}] Suspicious traffic detected: {features}\n"
                self.alert_text.insert(tk.END, alert_msg)
                self.alert_text.see(tk.END)
                total_alerts += 1
            else:
                traffic_msg = f"✔️ [{timestamp}] Normal traffic: {features}\n"
                self.traffic_text.insert(tk.END, traffic_msg)
                self.traffic_text.see(tk.END)
            
            # Update dashboard
            self.update_dashboard(timestamp, bytes_val, duration, packet_size)
            self.total_traffic_label.config(text=f"Total Traffic: {total_traffic}")
            self.alerts_label.config(text=f"Total Alerts: {total_alerts}")
            
            time.sleep(5)  # Update every 5 seconds

    def update_dashboard(self, timestamp, bytes_val, duration, packet_size):
        self.traffic_data['timestamp'].append(timestamp)
        self.traffic_data['bytes'].append(bytes_val)
        self.traffic_data['duration'].append(duration)
        self.traffic_data['packet_size'].append(packet_size)

        # Keep only the last 20 data points
        if len(self.traffic_data['timestamp']) > 20:
            for key in self.traffic_data:
                self.traffic_data[key] = self.traffic_data[key][-20:]

        self.ax.clear()
        self.ax.plot(self.traffic_data['timestamp'], self.traffic_data['bytes'], label='Bytes')
        self.ax.plot(self.traffic_data['timestamp'], self.traffic_data['packet_size'], label='Packet Size')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Value')
        self.ax.set_title('Network Traffic Over Time')
        self.ax.legend()
        self.ax.tick_params(axis='x', rotation=45)
        self.fig.tight_layout()
        self.canvas.draw()

    def export_alerts(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                               filetypes=[("Text files", "*.txt"),
                                                          ("All files", "*.*")])
        if file_path:
            with open(file_path, 'w') as file:
                file.write(self.alert_text.get("1.0", tk.END))
            messagebox.showinfo("Export Successful",
                              f"Alerts have been exported to {file_path}")

    def update_threshold(self):
        try:
            new_threshold = float(self.threshold_entry.get())
            if 0 <= new_threshold <= 1:
                # Update the model's threshold (assuming the model has this capability)
                # self.model.set_threshold(new_threshold)  # You might need to implement this method
                messagebox.showinfo("Threshold Updated",
                                  f"Alert threshold has been updated to {new_threshold}")
            else:
                messagebox.showerror("Invalid Threshold",
                                   "Please enter a value between 0 and 1")
        except ValueError:
            messagebox.showerror("Invalid Input",
                               "Please enter a valid number for the threshold")

    def retrain_model(self):
        # In a real-world scenario, you would load new data here
        self.model, self.accuracy = self.train_model()
        self.accuracy_label.config(text=f"Model Accuracy: {self.accuracy:.2f}%")
        messagebox.showinfo("Model Retrained",
                          f"Model has been retrained. New accuracy: {self.accuracy:.2f}%")

    def run(self):
        self.root.mainloop()

def main():
    app = ModernUI()
    app.run()

if __name__ == "__main__":
    main()