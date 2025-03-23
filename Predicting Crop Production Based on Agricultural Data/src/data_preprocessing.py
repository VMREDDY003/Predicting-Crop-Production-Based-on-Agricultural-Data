import pandas as pd

def load_and_preprocess_data(file_path):
    # Load Dataset from Excel
    df = pd.read_excel(file_path, engine="openpyxl")
    print("Dataset Loaded Successfully!")

    # Selecting Relevant Columns
    df = df[['Area', 'Item', 'Year', 'Element', 'Value']]

    # Pivoting the Data for ML
    df_pivot = df.pivot_table(index=['Area', 'Item', 'Year'], 
                              columns='Element', 
                              values='Value', 
                              aggfunc='sum').reset_index()

    # Print columns to debug
    print("Columns after pivoting:", df_pivot.columns)

    # Selecting only relevant columns
    df_pivot = df_pivot[['Area', 'Item', 'Year', 'Area harvested', 'Yield', 'Production']]

    # Renaming columns for clarity
    df_pivot.columns = ['Area', 'Item', 'Year', 'Area_Harvested', 'Yield', 'Production']

    # Handling Missing Values
    df_pivot.dropna(inplace=True)

    # Convert Data Types
    df_pivot['Year'] = df_pivot['Year'].astype(int)
    df_pivot['Area_Harvested'] = df_pivot['Area_Harvested'].astype(float)
    df_pivot['Yield'] = df_pivot['Yield'].astype(float)
    df_pivot['Production'] = df_pivot['Production'].astype(float)

    return df_pivot