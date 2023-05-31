import os 
import sys
import numpy as np 
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

from src.components.data_ingestion import DataIngestion

if __name__=='__main__':
    obj= DataIngestion()
    obj.initiate_data_ingestion()

    