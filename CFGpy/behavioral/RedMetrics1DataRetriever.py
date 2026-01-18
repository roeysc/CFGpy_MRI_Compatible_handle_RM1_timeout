import json
import os
from io import StringIO
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, parse_qs
from datetime import datetime
import pandas as pd
import requests

from CFGpy.behavioral import Configuration, DataRetriever
from CFGpy.behavioral._consts import (DATA_RETRIEVER_OUTPUT_FILENAME, CONFIG_URL_MISMATCH_ERROR)

try:
    from CFGpy.utils._nas_path import get_nas_path
except ImportError:
    def get_nas_path():
        return ""


class RedMetrics1DataRetriever(DataRetriever):
    def __init__(self, *, game_name: str | None = None, game_id: str | None = None,
                 game_version_ids: list[str] | None = None,
                 output_filename: str = DATA_RETRIEVER_OUTPUT_FILENAME,
                 config: Configuration = None, csv_directory: str = None,
                 input_url: str = None) -> None:
        super().__init__(game_name=game_name, game_id=game_id, output_filename=output_filename,
                         config=config if config is not None else Configuration.default(is_rm1=True))

        self._game_version_ids = self._config.GAME_VERSION_IDS or game_version_ids

        if input_url:
            print(f"Initializing retrieval from URL: {input_url}...")
            downloaded_meta = self._download_and_cache(input_url)
            self._csv_directory = Path(downloaded_meta['csv_directory'])

            if not self._game_id:
                self._game_id = downloaded_meta['game_id']
                self._config.GAME_ID = self._game_id
            if not self._game_name:
                self._game_name = downloaded_meta['game_name']
                self._config.GAME_NAME = self._game_name

        elif csv_directory:
            self._csv_directory = Path(csv_directory)
        else:
            try:
                self.nas_path = get_nas_path()
                self.csv_path = os.path.join(self.nas_path, "Projects", "CFG", "all_data_from_aws", "redmetrics")
                self._csv_directory = Path(self.csv_path)
            except Exception as e:
                raise FileNotFoundError(f"Could not determine NAS path. Error: {e}")

        if not self._csv_directory.exists():
            raise FileNotFoundError(f"The CSV directory does not exist: {self._csv_directory}")

        self._validate_input(input=[game_id, game_name, self._config.GAME_ID, self._config.GAME_NAME])
        self._validate_config()
        self._load_csv_files()

    def _download_and_cache(self, url: str, target_directory: str = "downloaded_data_cache") -> dict:
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        os.makedirs(target_directory, exist_ok=True)
        events_path = os.path.join(target_directory, "events.csv")

        # 1. Seamless Chunked Download
        after_str = query_params.get('after', ["2021-01-01T00:00:00.000Z"])[0]
        current_after_dt = pd.to_datetime(after_str.replace('Z', ''))
        end_dt = datetime.now()
        base_url_no_params = url.split('?')[0]
        game_id_from_url = query_params.get('game', [None])[0]

        print(f"Downloading 5-year data in 30-day chunks...")
        chunk_dfs = []

        while current_after_dt < end_dt:
            next_before_dt = current_after_dt + pd.Timedelta(days=30)
            if next_before_dt > end_dt: next_before_dt = end_dt

            after_val = current_after_dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            before_val = next_before_dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            chunk_url = f"{base_url_no_params}?game={game_id_from_url}&entityType=event&after={after_val}&before={before_val}"

            try:
                res = requests.get(chunk_url, timeout=(10, 120))
                res.raise_for_status()
                # Use 'skip' for bad lines (like the 1dd line)
                tmp = pd.read_csv(StringIO(res.text), on_bad_lines='skip')
                if not tmp.empty:
                    # Low threshold (3) to keep 'selected shape' events safe
                    chunk_dfs.append(tmp.dropna(thresh=3))
            except Exception as e:
                print(f"Chunk failed ({after_val[:10]}): {e}")

            current_after_dt = next_before_dt

        df_events = pd.concat(chunk_dfs, ignore_index=True).drop_duplicates()

        p_col = next((c for c in ['playerId', 'player_id', 'player'] if c in df_events.columns), None)
        if p_col:
            df_events[p_col] = df_events[p_col].astype(str).str.lstrip('0').replace('', '0')

        # Save the master file
        df_events.to_csv(events_path, index=False)

        # 2. METADATA SYNTHESIS (The Key to the 56 Subjects)
        game_id = query_params.get('game', [None])[0]

        # A. Find Version IDs actually present in the data
        v_col = next((c for c in ['gameVersion', 'version', 'gameVersion_id'] if c in df_events.columns), None)
        if v_col:
            actual_versions = df_events[v_col].unique()
            df_versions = pd.DataFrame({'id': actual_versions, 'game_id': game_id})
        else:
            df_versions = pd.DataFrame({'id': ['1.0'], 'game_id': game_id})
        df_versions.to_csv(os.path.join(target_directory, "game_versions.csv"), index=False)

        # B. Find Player IDs actually present in the data
        # --- Players Synthesis ---
        p_col = next((c for c in ['playerId', 'player_id', 'player'] if c in df_events.columns), None)
        if p_col:
            # 1. Convert to string
            # 2. Strip leading zeros so '092' becomes '92'
            # 3. This ensures '092' and '92' are treated as the same person
            df_events[p_col] = df_events[p_col].astype(str).str.lstrip('0')

            unique_p = df_events[p_col].unique()
            df_players = pd.DataFrame({'id': unique_p})
            df_players['name'] = df_players['id'].apply(lambda x: f"Player_{x}")
            df_players.to_csv(os.path.join(target_directory, "players.csv"), index=False)

        # C. Games metadata
        pd.DataFrame([{'id': game_id, 'name': f"Game_{game_id}"}]).to_csv(
            os.path.join(target_directory, "games.csv"), index=False)

        print(
            f"Successfully cached data. Subjects found: {len(unique_p) if p_col else 0}, Versions found: {df_versions['id'].tolist()}")

        return {"game_name": f"Game_{game_id}", "game_id": game_id, "csv_directory": target_directory}

    def _load_csv_files(self):
        """Load and normalize columns, specifically fixing the '092' vs '92' issue."""
        self._events_df = pd.read_csv(self._csv_directory / "events.csv")
        self._players_df = pd.read_csv(self._csv_directory / "players.csv")
        self._games_df = pd.read_csv(self._csv_directory / "games.csv")
        self._game_versions_df = pd.read_csv(self._csv_directory / "game_versions.csv")

        # Normalize Player ID Column Name
        if 'player_id' not in self._events_df.columns:
            p_col = next((c for c in ['playerId', 'player'] if c in self._events_df.columns), None)
            if p_col:
                self._events_df.rename(columns={p_col: 'player_id'}, inplace=True)

        # --- CRITICAL ID NORMALIZATION ---
        # Strip leading zeros from all player ID columns in all dataframes
        for df, col in [(self._events_df, 'player_id'), (self._players_df, 'id')]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lstrip('0')
                # Handle the edge case where '000' might become ''
                df[col] = df[col].replace('', '0')

        # Normalize Version Column Name
        if 'gameVersion_id' not in self._events_df.columns:
            v_col = next((c for c in ['gameVersion', 'version'] if c in self._events_df.columns), None)
            if v_col:
                self._events_df.rename(columns={v_col: 'gameVersion_id'}, inplace=True)

        # Ensure version IDs are strings for the filter
        self._events_df['gameVersion_id'] = self._events_df['gameVersion_id'].astype(str)
        self._game_versions_df['id'] = self._game_versions_df['id'].astype(str)

    def retrieve_data(self, *, verbose: bool = False, after: str = None, before: str = None,
                      event_type: str = None, section: str = None) -> pd.DataFrame:

        if verbose:
            print("Fetching all data from events.csv...")

        # process the loaded DF
        self._retrieved_df = self.fetch_all_data(verbose=verbose, after=after, before=before,
                                                 event_type=event_type, section=section)
        return self._format_df(verbose=verbose)

    def _validate_config(self) -> None:
        if not self._config.is_rm1:
            raise ValueError(CONFIG_URL_MISMATCH_ERROR)
        return None

    def get_game_version_ids(self):
        """Determine game version IDs based on provided inputs"""
        if self._game_version_ids:
            return self._game_version_ids

        elif self._game_id:
            matching_versions = self._game_versions_df[self._game_versions_df['game_id'] == self._game_id]
            if not matching_versions.empty:
                return matching_versions['id'].tolist()
            else:
                raise ValueError(f"No game_versions found for game_id: {self._game_id}")

        else:
            merged_df = self._game_versions_df.merge(
                self._games_df,
                left_on='game_id',
                right_on='id',
                suffixes=('', '_game')
            )
            matching_versions = merged_df[merged_df['name_game'] == self._game_name]
            if not matching_versions.empty:
                return matching_versions['id'].tolist()
            else:
                raise ValueError(f"No game_versions found for game name: {self._game_name}")

    def parse_json_column(self, *, df: pd.DataFrame, column_name: str, prefix: str):
        """Parse JSON columns in the DataFrame"""

        def try_parse(val):
            if pd.isna(val):
                return {}
            try:
                return json.loads(val)
            except json.JSONDecodeError:
                return {}

        parsed_df = df[column_name].apply(try_parse).apply(pd.Series)
        parsed_df.columns = [f"{prefix}.{col}" for col in parsed_df.columns]

        return pd.concat([df.drop(columns=[column_name]), parsed_df], axis=1)

    def convert_to_iso8601_millis(self, *, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Convert specified datetime columns to ISO 8601 format with milliseconds"""
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce") \
                              .dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ') \
                              .str.slice(stop=-4) + 'Z'
        return df

    def _format_df(self, *, verbose: bool = False) -> pd.DataFrame:
        """Format the retrieved DataFrame and FORCE player ID normalization."""
        if not self._retrieved_df.empty:
            if verbose:
                print("Formatting dataframe...")

            # 1. Standard RedMetrics 1 Renames
            self._retrieved_df.rename(columns={
                "gameVersion_id": "gameVersion",
                "player_id": "playerId",
                "birthDate": "playerBirthdate",
                "region": "playerRegion",
                "country": "playerCountry",
                "gender": "playerGender",
                "externalId": "playerExternalId",
            }, inplace=True)

            # 2. THE ID NORMALIZATION HAMMER
            # Strip leading zeros so subjects are consistent across all stages.
            id_cols = ['playerId', 'player_id', 'playerExternalId', 'id']
            for col in id_cols:
                if col in self._retrieved_df.columns:
                    self._retrieved_df[col] = (
                        self._retrieved_df[col]
                        .astype(str)
                        .str.lstrip('0')
                        .replace('', '0')
                    )

            # 3. Parse JSON custom data
            if "eventCustomData" in self._retrieved_df.columns:
                self._retrieved_df = self.parse_json_column(
                    df=self._retrieved_df,
                    column_name="eventCustomData",
                    prefix="customData"
                )

            # 4. Standardize Time Format
            self._retrieved_df = self.convert_to_iso8601_millis(
                df=self._retrieved_df,
                columns=["serverTime", "userTime"]
            )

        # 5. Ensure field order
        self._extra_fields = set(self._retrieved_df.columns) - set(self._config.DOWNLOADER_FIELD_ORDER)
        all_fields = self._config.DOWNLOADER_FIELD_ORDER + tuple(self._extra_fields)
        self._retrieved_df = self._retrieved_df.reindex(columns=all_fields)

        return self._retrieved_df

    def _create_df(self, *, game_version_id: str, after: str = None, before: str = None,
                   event_type: str = None, section: str = None) -> pd.DataFrame:

        # 1. Standard Filtering
        filtered_df = self._events_df[self._events_df['gameVersion_id'] == game_version_id].copy()
        if after: filtered_df = filtered_df[filtered_df['serverTime'] >= after]
        if before: filtered_df = filtered_df[filtered_df['serverTime'] <= before]
        if event_type: filtered_df = filtered_df[filtered_df['type'] == event_type]
        if section: filtered_df = filtered_df[filtered_df['section'].str.contains(section, na=False)]

        result_df = filtered_df.merge(self._players_df, left_on='player_id', right_on='id',
                                      suffixes=('', '_player'))

        # 2. UNMASKING LOGIC (Recover Real IDs)
        # Determine column names dynamically to handle both old and new data structures
        custom_data_col = 'customData_player' if 'customData_player' in result_df.columns else 'playerCustomData'
        external_id_col = 'externalId_player' if 'externalId_player' in result_df.columns else 'playerExternalId'
        if 'playerExternalId' not in result_df.columns and 'externalId' in result_df.columns:
            external_id_col = 'externalId'

        # Helper function to extract ID
        def extract_real_id(row):
            current_id = str(row.get(external_id_col, ''))
            # Check for masking
            if current_id.lower() == 'xxxxxx' or current_id == '' or current_id == 'nan' or current_id == 'None':
                try:
                    custom_data = json.loads(row.get(custom_data_col, '{}'))
                    return custom_data.get('userProvidedId', current_id)
                except:
                    return current_id
            return current_id

        # Apply Unmasking
        if external_id_col in result_df.columns and custom_data_col in result_df.columns:
            if not result_df.empty and 'xxxxxx' in result_df[external_id_col].astype(str).str.lower().values:
                print("Attempting to recover masked IDs from playerCustomData...")
                # We overwrite the column with the REAL ID
                result_df[external_id_col] = result_df.apply(extract_real_id, axis=1)

        # Now that IDs are unmasked, we filter them out immediately using the config, if needed
        if self._config and self._config.MANUALLY_EXCLUDED_IDS:
            # Ensure we are looking at the correct column for filtering
            target_col = external_id_col

            # Count before dropping
            initial_count = len(result_df)

            # Filter: Keep rows where ID is NOT in the excluded list
            # We convert to string to ensure "190" matches 190
            result_df = result_df[~result_df[target_col].astype(str).isin(self._config.MANUALLY_EXCLUDED_IDS)]

            dropped_count = initial_count - len(result_df)
            if dropped_count > 0:
                print(f"Exclusion applied: Dropped {dropped_count} rows matching excluded IDs.")

        # 4. Standard Cleanup
        if 'id_player' in result_df.columns: result_df = result_df.drop(columns=['id_player'])
        if 'customData' in result_df.columns: result_df = result_df.rename(
            columns={'customData': 'eventCustomData'})
        if 'customData_player' in result_df.columns: result_df = result_df.rename(
            columns={'customData_player': 'playerCustomData'})

        # Normalize the ID column name to what _format_df expects ('playerExternalId')
        if external_id_col != 'playerExternalId' and external_id_col in result_df.columns:
            result_df.rename(columns={external_id_col: 'playerExternalId'}, inplace=True)

        return result_df

    def fetch_all_data(self, *, verbose: bool = False, after: str = None, before: str = None,
                       event_type: str = None, section: str = None) -> pd.DataFrame:
        """Fetch all data from local CSV files (Iterates through versions)"""
        all_dfs = []
        game_version_ids = self.get_game_version_ids()

        for game_version_id in game_version_ids:
            if verbose:
                print(f"Fetching game_version_id={game_version_id}...")

            df = self._create_df(
                game_version_id=game_version_id,
                after=after,
                before=before,
                event_type=event_type,
                section=section
            )

            if not df.empty:
                all_dfs.append(df)

        result_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

        if verbose:
            print(f"Fetched {len(result_df)} rows total from {len(game_version_ids)} game version(s).")

        return result_df
