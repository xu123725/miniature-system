import os
import pandas as pd
import streamlit as st
from typing import Union, Optional
from .logger import logger

# Default encoding attempts
ENCODING_ORDER = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'latin1']

@st.cache_data(show_spinner=False)
def load_data(file_source: Union[str, st.runtime.uploaded_file_manager.UploadedFile]) -> pd.DataFrame:
    """
    Load meteorology data from a file path or Streamlit UploadedFile object.
    
    Args:
        file_source: A file path (str) or a Streamlit UploadedFile object.
        
    Returns:
        pd.DataFrame: The loaded data.
        
    Raises:
        ValueError: If the file cannot be read or decoded.
    """
    try:
        # Determine file name and extension
        if isinstance(file_source, str):
            if not os.path.exists(file_source):
                raise ValueError(f"File does not exist: {file_source}")
            file_name = file_source
        else:
            # Assume it's a Streamlit UploadedFile
            file_name = file_source.name
            # Reset pointer to start just in case
            file_source.seek(0)

        file_ext = os.path.splitext(file_name)[1].lower()
        
        # Determine delimiter
        delimiter = '|' if file_ext == '.nsv' else ','

        # Try multiple encodings
        df = None
        last_error = None

        for encoding in ENCODING_ORDER:
            try:
                # If it's an UploadedFile, we need to reset the pointer for each retry
                if not isinstance(file_source, str):
                    file_source.seek(0)
                
                df = pd.read_csv(file_source, encoding=encoding, delimiter=delimiter)
                logger.info(f"Successfully loaded data using {encoding} encoding.")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                last_error = e
                logger.debug(f"Failed with encoding {encoding}: {e}")
                continue
        
        if df is None:
            raise ValueError(f"Failed to decode file. Last error: {last_error}")

        # Clean column names
        df.columns = [str(c).strip() for c in df.columns]
        
        if df.empty:
            raise ValueError("The loaded file contains no data.")

        logger.info(f"Loaded {len(df)} records with columns: {list(df.columns)}")
        return df

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise ValueError(str(e))

def export_data_to_csv(data: Union[pd.DataFrame, list], filename: str) -> str:
    """
    Export data to CSV.
    """
    try:
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data
        
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        logger.info(f"Data exported to {filename}")
        return filename
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise e
