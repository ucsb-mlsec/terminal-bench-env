#!/usr/bin/env python3

"""
Table Catalog Interface

This module provides a catalog interface for managing table metadata.
The TableCatalog class acts as a facade to the underlying metadata storage system,
providing high-level operations for table registration and discovery.

The catalog implementation needs to interact with a MetadataStore backend
that handles the actual file I/O operations.
"""


class TableCatalog:
    """
    Catalog interface for table metadata management.
    
    This class provides methods to list, retrieve, register, and remove
    table metadata entries. It delegates storage operations to a MetadataStore
    instance.
    """
    
    def __init__(self, metadata_store):
        """
        Initialize the catalog with a metadata storage backend.
        
        Parameters:
            metadata_store: An instance of MetadataStore that handles
                          file operations for table metadata.
        """
        self.store = metadata_store
    
    def list_tables(self):
        """
        Returns a list of all registered table names.
        
        Should use self.store.list_tables() to retrieve the list.
        
        Returns:
            list: A list of table names (strings) currently registered
                 in the catalog.
        """
        # TODO: implement
        pass
    
    def get_table_metadata(self, table_name):
        """
        Retrieves metadata for a specific table.
        
        Parameters:
            table_name (str): The name of the table to retrieve metadata for.
        
        Returns:
            dict: A dictionary of metadata key-value pairs, or None if the
                 table is not found.
        
        Should use self.store.read_metadata(table_name).
        """
        # TODO: implement
        pass
    
    def register_table(self, table_name, metadata):
        """
        Registers a new table with its metadata.
        
        Parameters:
            table_name (str): The name of the table to register.
            metadata (dict): A dictionary containing the table's metadata
                           as key-value pairs.
        
        Should use self.store.write_metadata(table_name, metadata) to save
        the metadata.
        """
        # TODO: implement
        pass
    
    def remove_table(self, table_name):
        """
        Removes a table registration.
        
        Parameters:
            table_name (str): The name of the table to remove from the catalog.
        
        Should use self.store.delete_metadata(table_name) to remove the
        metadata file.
        """
        # TODO: implement
        pass