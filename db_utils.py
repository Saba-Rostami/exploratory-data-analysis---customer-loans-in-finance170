import yaml
from sqlalchemy import create_engine
import pandas as pd

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
        # Step 9: Load data from a .csv file into a Pandas DataFrame
        try:
            data_df = pd.read_csv(file_path)
            print(f"Data loaded from {file_path}")
            return data_df
        except Exception as e:
            print(f"Error: Unable to load data from {file_path}\n{str(e)}")
            return None
        


if __name__ == "__main__":
    # Create an instance of RDSDatabaseConnector
    rds_connector = RDSDatabaseConnector()

    # Extract data from the loan_payments table
    df = rds_connector.extract_data()

    # Display the extracted data
    df.head()
