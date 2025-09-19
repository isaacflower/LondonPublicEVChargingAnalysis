import pandas as pd
import numpy as np
    
class LADVehicleRegistrationDataProcessor:
    def __init__(self):
        self.df_veh0105_raw = None
        self.df_veh0142_raw = None
        self.df_veh0105 = None
        self.df_veh0142 = None
        self.lad_v_df = None
        self.lad_ev_df = None
    
    # === Core Methods ===

    def load_data(self, raw_data_paths: dict):
        """
        Load raw CSV data for vehicle types.
        
        Parameters:
        - raw_data_paths (dict): Dictionary with keys 'v' and/or 'ev' and file paths as values.
        """
        for vehicle_type, path in raw_data_paths.items():
            df_raw = pd.read_csv(path)
            if vehicle_type == 'v':
                self.df_veh0105_raw = df_raw
            elif vehicle_type == 'ev':
                self.df_veh0142_raw = df_raw

    def process_data(self, queries: dict[str], lad_list: np.ndarray, first_num_col: int = 3):
        """
        Preprocess, filter, and transform raw data to LAD-level time series.

        Parameters:
        - querys (dict[str]): Query strings to filter vehicle types (e.g., 'Fuel == "Petrol"').
        - lad_list (list): List of LAD names to include in the analysis.
        """
        self.df_veh0105 = self._preprocess(self.df_veh0105_raw, is_ev=False)
        self.df_veh0142 = self._preprocess(self.df_veh0142_raw, is_ev=True)

        df_v = self._transform_to_timeseries(self.df_veh0105, queries['v'], first_num_col)
        df_ev = self._transform_to_timeseries(self.df_veh0142, queries['ev'], first_num_col)

        self.lad_v_df = self._extract_lad_timeseries(df_v, lad_list)
        self.lad_ev_df = self._extract_lad_timeseries(df_ev, lad_list)

        self.lad_evms_df = self._calculate_evms(self.lad_ev_df, self.lad_v_df)

    def save_data(self):
        pass

    # === Internal Helper Methods === 

    def _preprocess(self, df: pd.DataFrame, is_ev: bool) -> pd.DataFrame:
        """
        Clean and standardize raw dataframe.
        
        Parameters:
        - df (pd.DataFrame): Raw input dataframe.
        - is_ev (bool): True if EV dataset, False if all vehicles.
        
        Returns:
        - pd.DataFrame: Cleaned dataframe.
        """
        col_map = {
            'Fuel [note 2]': 'Fuel',
            'Keepership [note 3]': 'Keepership',
            'ONS Geography [note 6]': 'Geography'
        }

        if is_ev:
            col_map.pop('Fuel [note 2]', None)

        df = df.rename(columns=col_map)
        df['Geography'] = df['Geography'].str.lstrip()

        df.replace(['[x]', '[low]'], np.nan, inplace=True)
        df.iloc[:, 7:] = df.iloc[:, 7:].replace(',', '', regex=True).astype(float)

        if not is_ev:
            df.iloc[:, 7:] *= 1000  # scale to actual vehicle count

        drop_cols = ['Units', 'ONS Sort [note 6]', 'ONS Code [note 6]']
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

        return df
    
    def _transform_to_timeseries(self, df: pd.DataFrame, query: str, first_num_col: int = 3) -> pd.DataFrame:
        """
        Apply query, pivot to time series format, and filter for Q4 values.

        Parameters:
        - df (pd.DataFrame): Preprocessed dataframe.
        - query (str): Query string to filter rows.
        - first_num_col (int): Index of the first numeric column of interest.

        Returns:
        - pd.DataFrame: Time series dataframe with years as index and LADs as columns.
        """
        df = df.query(query).iloc[:, first_num_col:]
        df.set_index('Geography', inplace=True)
        df = df.T[df.T.index.str.contains('Q4')]
        df.index = df.index.str[:4].astype(int)
        return df.astype(float)

    def _extract_lad_timeseries(self, df: pd.DataFrame, lad_list: list) -> pd.DataFrame:
        """
        Filter the time series dataframe for selected LADs from 2011 onwards.

        Parameters:
        - df (pd.DataFrame): Time series dataframe.
        - lad_list (list): List of LAD names to retain.

        Returns:
        - pd.DataFrame: Filtered LAD-level time series.
        """
        lad_df = df[lad_list]
        return lad_df.iloc[::-1].loc[2011:]

    def _calculate_evms(self, lad_ev_df: pd.DataFrame, lad_v_df: pd.DataFrame):
        """
        Calculate EV market share (EVMS) as the ratio of EVs to total vehicles.

        Parameters:
        - lad_ev_df (pd.DataFrame): LAD-level EV time series.
        - lad_v_df (pd.DataFrame): LAD-level total vehicle time series.

        Returns:
        - pd.DataFrame: LAD-level EV market share time series.
        """
        evms_df = lad_ev_df / lad_v_df
        evms_df.columns.rename('LAD', inplace=True)
        return evms_df