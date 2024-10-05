import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your DataFrame
df2 = pd.read_csv('Table2.csv')  # Adjust the path to your data

# Drop specified columns
columns_to_drop = [
    'Term Source REF.1', 'Term Accession Number.1',
    'Term Source REF.7', 'Term Accession Number.7',
    'Term Source REF.8', 'Term Accession Number.8',
    'Term Source REF.11', 'Term Accession Number.11',
    'Term Source REF.5', 'Term Accession Number.5'
]
df2.drop(columns=columns_to_drop, inplace=True)

# Set page title
st.title('Data Visualizations')

# Visualization dictionary
visualizations = {
    'Characteristics[Organism]': 'count',
    'Characteristics[Sex]': 'pie',
    'Parameter Value[Sample Storage Temperature]': 'box',
    'Characteristics[Material Type]': 'count',
    'Parameter Value[Carcass Preservation Method]': 'hist',
}

# Loop through visualizations
for column, plot_type in visualizations.items():
    st.subheader(f'Visualization for {column}')
    plt.figure(figsize=(12, 6))

    if plot_type == 'count':
        sns.countplot(data=df2, x=column, order=df2[column].value_counts().index)
        plt.title(f'Count of {column}')
        plt.xticks(rotation=45)
        plt.ylabel('Count')

    elif plot_type == 'pie':
        df2[column].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
        plt.title(f'Distribution of {column}')
        plt.ylabel('')

    elif plot_type == 'box':
        sns.boxplot(data=df2, y=column)
        plt.title(f'Distribution of {column}')
        plt.ylabel(column)

    elif plot_type == 'hist':
        sns.histplot(df2[column].dropna(), bins=10, kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')

    st.pyplot(plt)

# Additional visualizations

# Heatmap of correlations
numeric_df = df2.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap of Correlations')
st.pyplot(plt)

# Pairplot for selected numerical columns
sns.pairplot(df2, vars=['Parameter Value[Sample Storage Temperature]', 'Factor Value[Duration]'])
st.pyplot(plt)

# 1. Bar Plot for Material Types
plt.figure(figsize=(12, 6))
sns.countplot(data=df2, x='Characteristics[Material Type]', order=df2['Characteristics[Material Type]'].value_counts().index)
plt.title('Count of Material Types')
plt.xticks(rotation=45)
st.pyplot(plt)

# 2. Heatmap of Sample Storage Temperature vs. Preservation Method
pivot_table = df2.pivot_table(values='Parameter Value[Sample Storage Temperature]', 
                               index='Parameter Value[Carcass Preservation Method]', 
                               aggfunc='mean')
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, annot=True, cmap='viridis')
plt.title('Heatmap of Storage Temperature by Preservation Method')
st.pyplot(plt)

# 3. Box Plot for Duration by Organism
plt.figure(figsize=(12, 6))
sns.boxplot(data=df2, x='Characteristics[Organism]', y='Factor Value[Duration]')
plt.title('Duration by Organism')
plt.xticks(rotation=45)
st.pyplot(plt)

# 4. Count of Sex by Organism
plt.figure(figsize=(12, 6))
sns.countplot(data=df2, x='Characteristics[Organism]', hue='Characteristics[Sex]')
plt.title('Count of Organism by Sex')
plt.xticks(rotation=45)
st.pyplot(plt)

# 5. KDE Plot for Sample Storage Temperature
plt.figure(figsize=(12, 6))
sns.kdeplot(data=df2, x='Parameter Value[Sample Storage Temperature]', fill=True)
plt.title('KDE of Sample Storage Temperature')
st.pyplot(plt)

# 6. Facet Grid for Material Type by Organism
g = sns.FacetGrid(df2, col="Characteristics[Material Type]", col_wrap=3, height=4)
g.map(sns.histplot, "Parameter Value[Sample Storage Temperature]", bins=10, kde=True)
g.add_legend()
st.pyplot(g)

# 7. Strip Plot for Duration by Preservation Method
plt.figure(figsize=(12, 6))
sns.stripplot(data=df2, x='Parameter Value[Carcass Preservation Method]', y='Factor Value[Duration]', jitter=True)
plt.title('Duration by Carcass Preservation Method')
plt.xticks(rotation=45)
st.pyplot(plt)
