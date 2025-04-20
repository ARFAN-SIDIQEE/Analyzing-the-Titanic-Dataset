Titanic dataset analysis means studying the information about the passengers who were on the Titanic ship to find patterns — like who survived and who didn’t. The dataset includes details like age, gender, ticket class, and whether each person survived.
# Analyzing-the-Titanic-Dataset
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

df = pd.read_csv('tested.csv')

# Display basic info
print(df.info())

class TitanicAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Titanic Dataset Analyzer")
        self.root.geometry("1000x800")
        
        # Initialize variables
        self.df = None
        self.current_canvas = None
        
        # Create UI elements
        self.create_widgets()
    
    def create_widgets(self):
        # Frame for buttons
        button_frame = Frame(self.root)
        button_frame.pack(pady=10)
        
        # Buttons
        Button(button_frame, text="Load Dataset", command=self.load_data).pack(side=LEFT, padx=5)
        Button(button_frame, text="Show Gender Distribution", command=self.plot_gender_distribution).pack(side=LEFT, padx=5)
        Button(button_frame, text="Show Survival by Gender", command=self.plot_survival_by_gender).pack(side=LEFT, padx=5)
        Button(button_frame, text="Show Survival by Class", command=self.plot_survival_by_class).pack(side=LEFT, padx=5)
        Button(button_frame, text="Show Age Distribution", command=self.plot_age_distribution).pack(side=LEFT, padx=5)
        
        # Frame for displaying data
        self.display_frame = Frame(self.root)
        self.display_frame.pack(fill=BOTH, expand=True)
    
    def clear_display(self):
        """Clear previous plots and data displays"""
        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()
        for widget in self.display_frame.winfo_children():
            widget.destroy()
    
    def load_data(self):
        """Load and clean the dataset"""
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return
        
        try:
            self.df = pd.read_csv(file_path)
            
            # Clean data - handle null values
            self.df['Age'] = self.df['Age'].fillna(self.df['Age'].median())
            self.df['Embarked'] = self.df['Embarked'].fillna(self.df['Embarked'].mode()[0])
            self.df['Fare'] = self.df['Fare'].fillna(self.df['Fare'].median())
            
            # Create age groups
            bins = [0, 12, 18, 30, 50, 100]
            labels = ['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']
            self.df['AgeGroup'] = pd.cut(self.df['Age'], bins=bins, labels=labels)
            
            messagebox.showinfo("Success", "Dataset loaded and cleaned successfully!")
            
            # Show basic stats
            self.show_basic_stats()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset:\n{str(e)}")
    
    def show_basic_stats(self):
        """Display basic statistics about the dataset"""
        self.clear_display()
        
        if self.df is None:
            return
        
        stats_text = Text(self.display_frame, wrap=WORD)
        stats_text.pack(fill=BOTH, expand=True)
        
        # Basic statistics
        stats_text.insert(END, "=== Dataset Statistics ===\n\n")
        stats_text.insert(END, f"Total Passengers: {len(self.df)}\n\n")
        
        # Gender distribution
        gender_stats = self.df['Sex'].value_counts()
        stats_text.insert(END, "Gender Distribution:\n")
        stats_text.insert(END, gender_stats.to_string() + "\n\n")
        
        # Passenger class distribution
        class_stats = self.df['Pclass'].value_counts().sort_index()
        stats_text.insert(END, "Passenger Class Distribution:\n")
        stats_text.insert(END, class_stats.to_string() + "\n\n")
        
        # Survival rate
        survival_rate = self.df['Survived'].mean() * 100
        stats_text.insert(END, f"Overall Survival Rate: {survival_rate:.2f}%\n\n")
        
        # Make text read-only
        stats_text.config(state=DISABLED)
    
    def plot_gender_distribution(self):
        """Plot gender distribution"""
        if not self.check_data_loaded():
            return
        
        self.clear_display()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        self.df['Sex'].value_counts().plot(kind='bar', color=['pink', 'lightblue'], ax=ax)
        ax.set_title('Gender Distribution on Titanic')
        ax.set_xlabel('Gender')
        ax.set_ylabel('Number of Passengers')
        
        self.display_plot(fig)
    
    def plot_survival_by_gender(self):
        """Plot survival rate by gender"""
        if not self.check_data_loaded():
            return
        
        self.clear_display()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        survival_rates = self.df.groupby('Sex')['Survived'].mean() * 100
        survival_rates.plot(kind='bar', color=['pink', 'lightblue'], ax=ax)
        ax.set_title('Survival Rate by Gender')
        ax.set_xlabel('Gender')
        ax.set_ylabel('Survival Rate (%)')
        ax.set_ylim(0, 100)
        
        # Add value labels
        for i, v in enumerate(survival_rates):
            ax.text(i, v + 2, f"{v:.1f}%", ha='center')
        
        self.display_plot(fig)
    
    def plot_survival_by_class(self):
        """Plot survival rate by passenger class"""
        if not self.check_data_loaded():
            return
        
        self.clear_display()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        survival_rates = self.df.groupby('Pclass')['Survived'].mean() * 100
        survival_rates.plot(kind='bar', color=['gold', 'silver', 'brown'], ax=ax)
        ax.set_title('Survival Rate by Passenger Class')
        ax.set_xlabel('Class')
        ax.set_ylabel('Survival Rate (%)')
        ax.set_xticklabels(['1st Class', '2nd Class', '3rd Class'], rotation=0)
        ax.set_ylim(0, 100)
        
        # Add value labels
        for i, v in enumerate(survival_rates):
            ax.text(i, v + 2, f"{v:.1f}%", ha='center')
        
        self.display_plot(fig)
    
    def plot_age_distribution(self):
        """Plot age distribution with survival rates"""
        if not self.check_data_loaded():
            return
        
        self.clear_display()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Age distribution
        self.df['Age'].plot(kind='hist', bins=20, color='skyblue', ax=ax1)
        ax1.set_title('Age Distribution')
        ax1.set_xlabel('Age')
        ax1.set_ylabel('Number of Passengers')
        
        # Survival by age group
        survival_by_age = self.df.groupby('AgeGroup')['Survived'].mean() * 100
        survival_by_age.plot(kind='bar', color='lightgreen', ax=ax2)
        ax2.set_title('Survival Rate by Age Group')
        ax2.set_xlabel('Age Group')
        ax2.set_ylabel('Survival Rate (%)')
        ax2.set_ylim(0, 100)
        
        # Add value labels
        for i, v in enumerate(survival_by_age):
            ax2.text(i, v + 2, f"{v:.1f}%", ha='center')
        
        plt.tight_layout()
        self.display_plot(fig)
    
    def display_plot(self, figure):
        """Display matplotlib figure in Tkinter window"""
        self.current_canvas = FigureCanvasTkAgg(figure, master=self.display_frame)
        self.current_canvas.draw()
        self.current_canvas.get_tk_widget().pack(fill=BOTH, expand=True)
    
    def check_data_loaded(self):
        """Check if data is loaded before plotting"""
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a dataset first!")
            return False
        return True

# Main application
if __name__ == "__main__":
    root = Tk()
    app = TitanicAnalyzer(root)
    root.mainloop()
