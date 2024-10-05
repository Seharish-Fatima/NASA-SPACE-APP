import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

import streamlit as st

# Create a sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Visualizations"])

if page == "Home":
    st.title("Welcome to the Data Analysis App!")
    st.write("This is the home page.")

elif page == "Data Visualizations":
    import page2  # Import the second page


# Load the data
df1 = pd.read_csv("table1.csv")
df2 = pd.read_csv('Table2.csv')

# Drop unnecessary columns
columns_to_drop = [
    'Term Source REF.1', 'Term Accession Number.1', 'Term Source REF.2',
    'Term Accession Number.2', 'Term Source REF.3', 'Term Accession Number.3',
    'Term Source REF.4', 'Term Accession Number.4', 'Term Source REF.5',
    'Term Accession Number.5', 'Term Source REF.6', 'Term Accession Number.6',
    'Term Source REF.7', 'Term Accession Number.7', 'Term Source REF.8',
    'Term Accession Number.8', 'Protocol REF.1', 'Protocol REF.2',
    'Protocol REF.3', 'Protocol REF.4', 'Unit.1', 'Unit.2', 'Unit.3', 'Unit.4'
]
df1.drop(columns=columns_to_drop, inplace=True)

# Create a function to generate visualizations
def create_visualizations(df):
    st.title("Data Visualizations")
    
    # Add a sidebar for user interaction
    st.sidebar.header("Select Visualization Type")
    visualization_type = st.sidebar.selectbox("Choose a visualization:", [
        "Bar Plot", "Histogram", "Box Plot", "Scatter Plot", "Line Plot", 
        "Count Plot", "Violin Plot", "Heatmap", "Pair Plot", "Facet Grid", 
        "Swarm Plot", "Joint Plot", "KDE Plot", "Strip Plot", "Rug Plot"
    ])
    
    if visualization_type == "Bar Plot":
        st.subheader("Bar Plot: QA Score by Sample Name")
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Sample Name', y='Parameter Value[QA Score]', data=df)
        plt.xticks(rotation=90)
        st.pyplot(plt)

    elif visualization_type == "Histogram":
        st.subheader("Histogram of QA Scores")
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Parameter Value[QA Score]'], bins=10)
        st.pyplot(plt)

    elif visualization_type == "Box Plot":
        st.subheader("Box Plot: QA Score by Library Selection")
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Parameter Value[library selection]', y='Parameter Value[QA Score]', data=df)
        plt.xticks(rotation=90)
        st.pyplot(plt)

    elif visualization_type == "Scatter Plot":
        st.subheader("Scatter Plot: Read Length vs. rRNA Contamination")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Parameter Value[Read Length]', y='Parameter Value[rRNA Contamination]', data=df)
        st.pyplot(plt)

    elif visualization_type == "Line Plot":
        st.subheader("Line Plot: Read Length vs. rRNA Contamination")
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='Parameter Value[Read Length]', y='Parameter Value[rRNA Contamination]')
        st.pyplot(plt)

    elif visualization_type == "Count Plot":
        st.subheader("Count Plot: Stranded")
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Parameter Value[stranded]', data=df)
        st.pyplot(plt)

    elif visualization_type == "Violin Plot":
        st.subheader("Violin Plot: QA Score by Library Layout")
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='Parameter Value[library layout]', y='Parameter Value[QA Score]', data=df)
        plt.xticks(rotation=90)
        st.pyplot(plt)

    elif visualization_type == "Heatmap":
        st.subheader("Heatmap of Correlations")
        plt.figure(figsize=(10, 6))
        numeric_df = df.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
        sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
        st.pyplot(plt)

    elif visualization_type == "Pair Plot":
        st.subheader("Pair Plot")
        pair_plot = sns.pairplot(df, diag_kind='kde')
        st.pyplot(pair_plot.fig)

    elif visualization_type == "Facet Grid":
        st.subheader("Facet Grid: QA Score by Library Layout")
        g = sns.FacetGrid(df, col='Parameter Value[library layout]')
        g.map(sns.histplot, 'Parameter Value[QA Score]')
        st.pyplot(g.fig)

    elif visualization_type == "Swarm Plot":
        st.subheader("Swarm Plot: Read Length vs. rRNA Contamination")
        plt.figure(figsize=(10, 6))
        sns.swarmplot(x='Parameter Value[Read Length]', y='Parameter Value[rRNA Contamination]', data=df)
        st.pyplot(plt)

    elif visualization_type == "Joint Plot":
        st.subheader("Joint Plot: Read Length vs. rRNA Contamination")
        joint_plot = sns.jointplot(x='Parameter Value[Read Length]', y='Parameter Value[rRNA Contamination]', data=df)
        st.pyplot(joint_plot.fig)

    elif visualization_type == "KDE Plot":
        st.subheader("KDE Plot of QA Scores")
        plt.figure(figsize=(10, 6))
        sns.kdeplot(df['Parameter Value[QA Score]'], fill=True)
        st.pyplot(plt)

    elif visualization_type == "Strip Plot":
        st.subheader("Strip Plot: QA Score by Library Selection")
        plt.figure(figsize=(10, 6))
        sns.stripplot(x='Parameter Value[library selection]', y='Parameter Value[QA Score]', data=df)
        plt.xticks(rotation=90)
        st.pyplot(plt)

    elif visualization_type == "Rug Plot":
        st.subheader("Rug Plot of QA Scores")
        plt.figure(figsize=(10, 6))
        sns.rugplot(df['Parameter Value[QA Score]'])
        st.pyplot(plt)

# Call the function to create visualizations
create_visualizations(df1)


