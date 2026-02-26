import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

         # 1. Load Data
df = pd.read_csv('Titanic-Dataset.csv')

# 2. Set Theme and Figure Layout
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
plt.subplots_adjust(hspace=0.4, wspace=0.3) # Space between plots

# -- Quadrant (0,0): Bar Chart: Survival count by Gender
sns.countplot(data=df, x='Sex', hue='Survived', palette='Set1', ax=axes[0, 0])
axes[0, 0].set_title('Survival Count by Gender', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_xlabel('Gender')

# -- Quadrant (0,1): Histogram: Age Distribution
axes[0, 1].hist(df['Age'].dropna(), bins=30, color='skyblue', edgecolor='black')
axes[0, 1].set_title('Age Distribution of Passengers', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Age')
axes[0, 1].set_ylabel('Frequency')

# -- Quadrant (1,0): Pie Chart: Passenger Class Distribution
pclass_counts = df['Pclass'].value_counts().sort_index()
axes[1, 0].pie(pclass_counts, labels=['Class ' + str(idx) for idx in pclass_counts.index], 
               autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99'])
axes[1, 0].set_title('Passenger Class Distribution', fontsize=14, fontweight='bold')

# -- Quadrant (1,1): KDE Plot: Survival Density by Age
sns.kdeplot(data=df[df['Survived'] == 0], x='Age', fill=True, color="#e74c3c", 
            label="Did Not Survive", alpha=0.5, linewidth=2, ax=axes[1, 1])
sns.kdeplot(data=df[df['Survived'] == 1], x='Age', fill=True, color="#2ecc71", 
            label="Survived", alpha=0.5, linewidth=2, ax=axes[1, 1])
axes[1, 1].set_title('Survival Density by Age', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Passenger Age')
axes[1, 1].set_ylabel('Density (Probability)')
axes[1, 1].legend(title='Outcome')

# Global Title and Layout Adjustment
plt.suptitle('Titanic Dataset: Key Survival Insights', fontsize=20, fontweight='bold', y=0.96)
plt.savefig('titanic_quadrant_plot.png', dpi=300, bbox_inches='tight')
plt.show()
