import yaml
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import math
from scipy import stats
from scipy.stats import normaltest
from statsmodels.graphics.gofplots import qqplot

from sklearn.impute import KNNImputer

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import warnings

# To ignore all warnings
warnings.filterwarnings("ignore")


class RDSDatabaseConnector:
    def __init__(self, credentials_path='credentials.yaml'):
        # Load credentials
        self.credentials = self.load_credentials(credentials_path)

        # Initialize the SQLAlchemy engine
        self.engine = self.init_engine()

    def load_credentials(self, file_path):
        # Load credentials from YAML file
        with open(file_path, 'r') as file:
            credentials = yaml.safe_load(file)
        return credentials

    def init_engine(self):
    # Initialize SQLAlchemy engine
        try:
            engine = create_engine(
                f"postgresql://{self.credentials['RDS_USER']}:{self.credentials['RDS_PASSWORD']}@"
                f"{self.credentials['RDS_HOST']}:{self.credentials['RDS_PORT']}/{self.credentials['RDS_DATABASE']}"
                )
            
            print("Connected to the database")
            print(f"Database URL: {engine.url}")
            return engine
        except Exception as e:
            print(f"Error: Unable to connect to the database\n{str(e)}")
            return None


    def extract_data(self, table_name='loan_payments'):
        # Extract data from the RDS database
        try:
            query = f"SELECT * FROM {table_name}"

            # Execute the query and load data into a Pandas DataFrame
            data_df = pd.read_sql_query(query, self.engine)
            return data_df
        except Exception as e:
            print(f"Error: Unable to extract data from the database\n{str(e)}")
            return None
        
    def save_to_csv(self, data_df, file_path='loan_payments.csv'):
        # Save data to a .csv file
        try:
            data_df.to_csv(file_path, index=False)
            print(f"Data saved to {file_path}")
        except Exception as e:
            print(f"Error: Unable to save data to {file_path}\n{str(e)}")
        

    def load_from_csv(self, file_path='loan_payments.csv'):
        # Load data from a .csv file into a Pandas DataFrame
        try:
            data_df = pd.read_csv(file_path)
            print(f"Data loaded from {file_path}")
            return data_df
        except Exception as e:
            print(f"Error: Unable to load data from {file_path}\n{str(e)}")
            return None
        

class DataTransform:
    def __init__(self, data):
        self.data = data.copy()

    def convert_to_numeric(self, column_name):
        # Convert a column to numeric format.
        self.data[column_name] = pd.to_numeric(self.data[column_name], errors='coerce')
        return self.data[column_name]
    
    def convert_to_integer(self, column_name):
        try:
            # Attempt to convert the column to integer
            self.data[column_name] = self.data[column_name].astype("int")
        except ValueError as e:
            print(f"Error converting column {column_name} to integer: {e}")
            # Handle the error or consider cleaning the column before conversion
            return self.data[column_name]

    def convert_to_datetime(self, column_name):
        # Convert a column to datetime format.
        self.data[column_name] = pd.to_datetime(self.data[column_name], errors='coerce')
        return self.data[column_name]

    def convert_to_categorical(self, column_name):
        # Convert a column to categorical format.
        self.data[column_name] = self.data[column_name].astype('category')
        return self.data[column_name]

    def remove_excess_symbols(self, column_name, symbols_to_remove):
        # Remove excess symbols from a column.
        self.data[column_name] = self.data[column_name].str.replace(symbols_to_remove, '')
        return self.data[column_name]

    def remove_non_digit_charactor(self,column_name):
        self.data[column_name] = self.data[column_name].str.replace(r'\D', '', regex=True)
        return self.data[column_name]
    
    def get_transformed_data(self):
        # Return the transformed DataFrame.
        return self.data


class DataFrameInfo:
    def __init__(self, df):
        self.df = df

    def describe_columns(self):
        # Describe all columns in the DataFrame to check their data types.
        return self.df.dtypes

    def extract_statistics(self):
        # Extract statistical values: median, standard deviation, and mean from the columns and the DataFrame.
        return self.df.describe().T

    def count_distinct_values(self):
        # Count distinct values in categorical columns.
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        distinct_values = {}
        for col in categorical_columns:
            distinct_values[col] = self.df[col].nunique()

        # Create a DataFrame from the dictionary
        distinct_values_df = pd.DataFrame(list(distinct_values.items()), columns=['Column', 'Distinct Values'])
        distinct_values_df = distinct_values_df.sort_values(by='Distinct Values', ascending=False)
        return distinct_values_df


    def print_shape(self):
        # Print out the shape of the DataFrame.
        return self.df.shape

    def count_null_values(self):
        # Generate a count/percentage count of NULL values in each column.
        null_counts = self.df.isnull().sum()
        null_percentage = (null_counts / len(self.df)) * 100
        null_info = pd.DataFrame({
            'Null Count': null_counts,
            'Null Percentage': null_percentage
        }).sort_values(by="Null Count", ascending=False)

        # Filter and display columns with null values
        null_info = null_info[null_info['Null Count'] > 0]
        if not null_info.empty:
            print("Columns with NULL values:")
            print(null_info)
        else:
            print("No NULL values found in any column.")

    def separate_columns(self):
        # Separate DataFrame columns into object and numeric
        object_columns = list(self.df.select_dtypes(include=['object']).columns)
        numeric_columns = list(self.df.select_dtypes(include=['number']).columns)

        print(f"We have {len(object_columns)} object columns in this dataset. \n Object Columns: \n {object_columns} \n")
        print("------------------------------------")
        print(f"We have {len(numeric_columns)} object columns in this dataset. \n Numeric Columns: \n {numeric_columns}\n") 

    def object_columns(self):
        object_columns = list(self.df.select_dtypes(include=['object']).columns)
        return object_columns

    def numeric_columns(self):
        numeric_columns = list(self.df.select_dtypes(include=['number']).columns)
        return numeric_columns
    
    def find_outliers(self, col):
        Q1 = self.df[col].describe()['25%']
        Q3 = self.df[col].describe()['75%']
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5*IQR
        upper_bound = Q3 + 1.5*IQR
        outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
        return outliers

        
class DataFrameTransform:
    def __init__(self, df):
        self.df = df

    def null_check(self):
        # Determine the amount of NULLs in each column
        null_counts = self.df.isnull().sum()
        print("NULL counts in each column:")
        print(null_counts)

        # Determine which columns to drop 
        columns_to_drop = null_counts[null_counts > 0.2 * len(self.df)].index

        # Drop columns with a high percentage of NULL values
        self.df = self.df.drop(columns=columns_to_drop)

    def impute_column(self, column_name):
        # Check if the column has NULL values
        if self.df[column_name].isnull().any():
            # Check if the column is numeric
            if pd.api.types.is_numeric_dtype(self.df[column_name]):
                # Decide whether to impute with mean or median based on skewness
                skewness = self.df[column_name].skew()
                if abs(skewness) > 1:
                    impute_value = self.df[column_name].median()
                    imputation_method = 'median'
                else:
                    impute_value = self.df[column_name].mean()
                    imputation_method = 'mean'
            else:
                # Impute with mode for categorical data
                impute_value = self.df[column_name].mode().iloc[0]
                imputation_method = 'mode'
            
            # Impute the NULL values
            self.df[column_name].fillna(impute_value, inplace=True)
            
            print(f"Imputed '{column_name}' using {imputation_method}. Imputed value: {impute_value}")
        else:
            print(f"No NULL values found in '{column_name}'. No imputation performed.")


    def check_nulls_after_imputation(self):
        # Check if there are any remaining NULL values
        null_counts_after_imputation = self.df.isnull().sum()
        print("NULL counts after imputation:")
        print(null_counts_after_imputation)

    def identify_skewed_columns(self, threshold=0.2):
        self.df = self.df.select_dtypes(include=['number'])
        skewed_columns = self.df.apply(lambda x: x.skew()).abs() > threshold
        return skewed_columns.index[skewed_columns].tolist()
    
    def perform_normality_test(self, column_name):
        """
        Perform D’Agostino’s K^2 Test for normality on a specific column.

        Parameters:
        - column_name: str. The name of the column to be tested.

        Returns:
        - stat: Test statistic.
        - p: p-value.
        """
        
        data = self.df[column_name]
        stat, p = normaltest(data, nan_policy='omit')
        return stat, p
    
    def print_normality_test(self):
        
        numerical_columns = self.df.select_dtypes(include=['number']).columns

        for column_name in numerical_columns:
            data = self.df[column_name]
            stat, p = normaltest(data, nan_policy='omit')
            print(f'{column_name}: Statistics: {stat:.3f}, p-value: {p:.3f}')


    def log_transform_numeric_columns(self, columns):
        
        # transformed_df = self.df.copy()
        # Log transformation for specified numeric columns
        for col in columns:
            self.df[col] = np.log1p(self.df[col])
         
        return self.df
    
    def log_transform_single_column(self, column_name):
        
        # Log transformation for a single column
        self.df[column_name + '_log'] = np.log1p(self.df[column_name])
        return self.df


    def boxcox_transform_single_column(self, column_name):
        # Box-Cox transformation for a single column
        self.df[column_name + '_boxcox'] = stats.boxcox(self.df[column_name] + 1)  # Adding 1 to handle zero values
        return self.df

    def boxcox_transform_numerical_columns(self):
        """
        Perform Box-Cox transformation on numerical columns in the DataFrame.

        Returns:
        - transformed_df: New DataFrame with Box-Cox transformed columns.
        """
        columns = self.df.select_dtypes(include=['number']).columns
        transformed_df = self.df.copy()

        for col in columns:
            # Adding 1 to handle zero values
            transformed_data, lambda_param = stats.boxcox(self.df[col] + 1)
            transformed_df[col + '_boxcox'] = transformed_data

        return transformed_df
    
    def yeojohnson_transform_single_column(self, column_name):
        # Box-Cox transformation for a single column
        self.df[column_name + '_yeojohnson'] = stats.yeojohnson(self.df[col] + 1)  # Adding 1 to handle zero values
        return self.df

    def yeojohnson_transform_numeric_columns(self):
        columns = self.df.select_dtypes(include=['number']).columns
        transformed_df = self.df.copy()
        # Yeo-Johnson transformation for specified numeric columns
        for col in columns:
            transformed_data, lambda_param = stats.yeojohnson(self.df[col] + 1)  # Adding 1 to handle zero values
            transformed_df[col + '_yeojohnson'] = transformed_data
        return transformed_df
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        """
        Remove outliers from the DataFrame using the z-score method.

        Parameters:
        - columns (list): A list of column names to consider for outlier removal.
                          If None, all numeric columns will be used.
        - threshold (float): The z-score threshold for detecting outliers. Data points
                            with a z-score higher than this threshold will be considered outliers.
        """
        # Select columns for outlier removal
        if columns is None:
            numeric_columns = self.df.select_dtypes(include='number').columns
        else:
            numeric_columns = [col for col in columns if col in self.df.columns]

        if not numeric_columns:
            print("No valid numeric columns selected for outlier removal.")
            return

        # Calculate z-scores for each selected numeric column
        z_scores = pd.DataFrame()
        for column in numeric_columns:
            z_scores[column + '_zscore'] = (self.df[column] - self.df[column].mean()) / self.df[column].std()

        # Identify outliers based on the threshold
        outliers = (z_scores.abs() > threshold).any(axis=1)

        # Remove outliers from the DataFrame
        self.df = self.df[~outliers].reset_index(drop=True)

    def save_copy(self, filename):
        self.df.to_csv(filename, index=False)


    def remove_highly_correlated_columns(self, threshold=0.95, columns_to_exclude=None):
        # Create a correlation matrix
        corr_matrix = self.df.select_dtypes(include="number").corr().abs()

        # Exclude specified columns from the correlation analysis
        if columns_to_exclude:
            corr_matrix = corr_matrix.drop(columns=columns_to_exclude, index=columns_to_exclude, errors="ignore")

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find index of feature columns with correlation greater than the threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        print("Columns to be dropped: \n\n", to_drop, "\n\n")

        # Drop the highly correlated columns
        df_filtered = self.df.drop(to_drop, axis=1)

        return df_filtered



class KNNImputerClass:
    def __init__(self, k_neighbors=5):
        """
        Constructor for the KNNImputerClass.

        Parameters:
        - k_neighbors (int): Number of neighbors to use for imputation.
        """
        self.k_neighbors = k_neighbors
        self.knn_imputer = KNNImputer(n_neighbors=k_neighbors)

    def fit_transform(self, df):
        """
        Fit the KNN imputer on the given DataFrame and perform imputation.

        Parameters:
        - df (pd.DataFrame): DataFrame with missing values.

        Returns:
        - pd.DataFrame: DataFrame with missing values imputed.
        """
        # Ensure that the input DataFrame has missing values
        if df.isnull().sum().sum() == 0:
            raise ValueError("Input DataFrame does not have missing values.")

        # Perform k-nearest neighbors imputation
        imputed_data = self.knn_imputer.fit_transform(df)

        # Convert the imputed data array back to a DataFrame
        imputed_df = pd.DataFrame(imputed_data, columns=df.columns)

        return imputed_df

    

class Plotter:
    def __init__(self, df):
        self.df = df

    @staticmethod
    def visualize_null_removal(original_data, transformed_data):
        # Visualize the removal of NULL values
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        original_data.isnull().sum().plot(kind='bar', title='NULL values before removal', ax=ax[0])
        transformed_data.isnull().sum().plot(kind='bar', title='NULL values after removal', ax=ax[1])
        plt.show()

    def visualize_skewness_sns(self, columns):
        for column in columns:
            plt.figure(figsize=(8, 5))
            sns.histplot(self.df[column], kde=True)
            plt.title(f'Skewness of {column}')
            plt.show()

    def visualize_skewness_plotly(self, columns):
        for column in columns:
            # Create a histogram with Plotly Express
            fig = px.histogram(self.df, x=column, nbins=30, title=f'Skewness of {column}', labels={column: f'{column} Value'})
            fig.show()

    def plot_numeric_distribution(self):
        numeric_columns = self.df.select_dtypes(include=['number']).columns
        num_cols = len(numeric_columns)

        # Calculate the number of rows needed for two columns in each row
        num_rows = math.ceil(num_cols / 2)

        # Create subplots with two columns in each row
        fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))
        axes = axes.flatten()

        for i, col in enumerate(numeric_columns):
            # Plot distribution for each numeric column
            sns.histplot(self.df[col], kde=True, ax=axes[i], color='deeppink')  # Set color to deeppink
            axes[i].set_title(f'Distribution of {col}', pad=20)  # Added pad to move the title to the middle

        # Remove empty subplots
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def plot_categorical_columns(self):
        """
        Plot categorical columns.

        The function selects categorical columns automatically and creates two plots in each row.

        Colors: deep pink, violet, sky blue, light pink, purple, light blue, light green, light coral.

        """
        cat_cols = self.df.select_dtypes(include=['object']).columns
        categorical_columns = [x for x in cat_cols if len(self.df[x].value_counts().index) < 10]
        colors = ['Rose Pink', 'Lavender', 'Sky Blue', 'Mauve', 'Periwinkle', 
                  'Cerulean Blue', 'Fuchsia', 'Indigo', 'Baby Pink', 'Sapphire Blue']

        num_columns = len(categorical_columns)
        num_rows = (num_columns + 1) // 2  # Ensure at least 1 row

        # Adjust vertical spacing based on the number of rows
        vertical_spacing = min(0.1, 1 / num_rows)

        fig = make_subplots(rows=num_rows, cols=2, subplot_titles=categorical_columns,
                            horizontal_spacing=0.1, vertical_spacing=vertical_spacing)

        for i, column_name in enumerate(categorical_columns):
            row = (i // 2) + 1
            col = i % 2 + 1

            # Create bar plot
            fig.add_trace(
                px.bar(self.df, x=column_name, 
                       color_discrete_sequence=[colors[i % len(colors)]]).update_traces(
                           showlegend=False).data[0],
                row=row, col=col
            )

        fig.update_layout(title_text="Distribution of Categorical Columns", 
                          showlegend=False)

        # Update subplot titles to be in the middle
        for annotation in fig['layout']['annotations']:
            annotation.update(x=0.5, xanchor='center')

        fig.show()

    def plot_categorical_columns2(self):
        """
        Plot categorical columns.

        The function selects categorical columns automatically and creates two plots in each row.

        Colors: deep pink, violet, sky blue, light pink, purple, light blue, light green, light coral.

        """
        cat_cols = self.df.select_dtypes(include=['object']).columns
        categorical_columns = [x for x in cat_cols if len(self.df[x].value_counts().index) < 10]
        colors = ['Rose Pink', 'Lavender', 'Sky Blue', 'Mauve', 'Periwinkle', 
              'Cerulean Blue', 'Fuchsia', 'Indigo', 'Baby Pink', 'Sapphire Blue']

        num_columns = len(categorical_columns)
        num_rows = (num_columns + 1) // 2  # Ensure at least 1 row

        # Adjust vertical spacing based on the number of rows
        vertical_spacing = min(0.1, 1 / num_rows)

        fig = make_subplots(rows=num_rows, cols=2, subplot_titles=categorical_columns,
                        horizontal_spacing=0.1, vertical_spacing=vertical_spacing)

        for i, column_name in enumerate(categorical_columns):
            row = (i // 2) + 1
            col = i % 2 + 1

            # Create bar plot
            fig.add_trace(go.Bar(x=self.df[column_name].value_counts().index, 
                                 y=self.df[column_name].value_counts(),
                                 marker_color=colors[i % len(colors)]),
                                 row=row, col=col)

        fig.update_layout(title_text="Distribution of Categorical Columns", 
                          showlegend=False)

        # Update subplot titles to be in the middle
        for annotation in fig['layout']['annotations']:
            annotation.update(x=0.5, xanchor='center')

        fig.show()


    def plot_qqplots_numerical_columns(self):
        """
        Plot QQ plots for all numerical columns using Plotly Express.

        Two plots in each row, titles in the middle, and using specified colors.

        """
        numerical_columns = self.df.select_dtypes(include=['number']).columns
        colors = ['deeppink', 'violet']

        num_columns = len(numerical_columns)
        num_rows = (num_columns + 1) // 2  # Ensure at least 1 row

        fig = make_subplots(rows=num_rows, cols=2, 
                            subplot_titles=numerical_columns, 
                            horizontal_spacing=0.1, vertical_spacing=0.2)

        for i, column_name in enumerate(numerical_columns):
            row = (i // 2) + 1
            col = i % 2 + 1

            # Create QQ plot
            qq_plot = qqplot(self.df[column_name], scale=1, line='q', fit=True)
            qq_fig = px.mpl_to_plotly(qq_plot)

            # Update marker color
            qq_fig.update_traces(marker=dict(color=colors[i % len(colors)]))

            # Add QQ plot to subplots
            fig.add_trace(qq_fig.data[0], row=row, col=col)

        # Update layout
        fig.update_layout(title_text="QQ Plots for Numerical Columns", showlegend=False)

        # Update subplot titles to be in the middle
        for annotation in fig['layout']['annotations']:
            annotation.update(x=0.5, xanchor='center')

        fig.show()

    def plot_pairwise_correlation(self, cols_to_drop, title=None):
        correlation_matrix = self.df.select_dtypes(exclude="category").drop(columns=cols_to_drop).corr().round(2)

        # Create a mask to display only the lower triangle of the correlation matrix
        mask = np.tril(np.ones_like(correlation_matrix, dtype=bool))

        # Set the upper triangle values to NaN to hide them in the plot
        correlation_matrix_lower = correlation_matrix.where(mask)

        fig = px.imshow(correlation_matrix_lower, 
                        text_auto=True)
        fig.update_xaxes(showline=False)
        fig.update_yaxes(showline=False)

        default_title = "Pairwise Correlation of Columns"
        fig.update_layout(
            title={'text': title or default_title,
                   'x': 0.50,
                   'xanchor': 'center',
                   'yanchor': 'top',
                   'font': {'size': 20}},
            margin={'t': 100},
            width=1000, height=1000
        )
        fig.show()



    

    





        
'''

if __name__ == "__main__":
    # Create an instance of RDSDatabaseConnector
    connector = RDSDatabaseConnector()

    # Extract data from the loan_payments table
    df = connector.extract_data()

    # Display the extracted data
    df.head()

'''