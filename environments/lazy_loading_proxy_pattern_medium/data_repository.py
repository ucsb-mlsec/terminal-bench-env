import pandas as pd
import os


class DataRepository:
    def __init__(self, data_dir):
        """
        Initialize the DataRepository with lazy loading.
        
        Args:
            data_dir: Path to directory containing CSV files
        """
        self.data_dir = data_dir
        self._customers_data = None
        self._products_data = None
        self._orders_data = None
        self._transactions_data = None
    
    def get_customers(self):
        """
        Get customers data, loading it on first access.
        
        Returns:
            pandas DataFrame containing customers data
        """
        if self._customers_data is None:
            self._customers_data = pd.read_csv(os.path.join(self.data_dir, 'customers.csv'))
        return self._customers_data
    
    def get_products(self):
        """
        Get products data, loading it on first access.
        
        Returns:
            pandas DataFrame containing products data
        """
        if self._products_data is None:
            self._products_data = pd.read_csv(os.path.join(self.data_dir, 'products.csv'))
        return self._products_data
    
    def get_orders(self):
        """
        Get orders data, loading it on first access.
        
        Returns:
            pandas DataFrame containing orders data
        """
        if self._orders_data is None:
            self._orders_data = pd.read_csv(os.path.join(self.data_dir, 'orders.csv'))
        return self._orders_data
    
    def get_transactions(self):
        """
        Get transactions data, loading it on first access.
        
        Returns:
            pandas DataFrame containing transactions data
        """
        if self._transactions_data is None:
            self._transactions_data = pd.read_csv(os.path.join(self.data_dir, 'transactions.csv'))
        return self._transactions_data