################################
# Creator: Filipy              #
# LinkedIn: Filipy S. Furtado  #
# Instagram: sf.filipy         #
# Discord: furth__             #
################################

import pandas as pd
import numpy as np

class kNN:
    def __init__():
        pass

    def getData(url, type_):
        try:
            if type_ == "csv":
                self.data = pd.read_csv(url)
            elif type_ == "excel":
                self.data = pd.read_csv(url)
            else:
                return FileNotFoundError

        except Exception as e:
            return e




