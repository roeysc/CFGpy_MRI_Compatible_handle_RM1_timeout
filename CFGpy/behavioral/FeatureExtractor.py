import numpy as np
import pandas as pd
from datetime import datetime
from CFGpy.behavioral.data_interfaces import PostparsedDataset
from CFGpy.behavioral._consts import (FEATURES_ID_KEY, FEATURES_START_TIME_KEY, N_CLUSTERS_KEY, GAME_DURATION_KEY,
                                      N_MOVES_KEY, LONGEST_PAUSE_KEY, MEDIAN_EXPLORE_LENGTH_KEY, N_GALLERIES_KEY,
                                      SELF_AVOIDANCE_KEY, EXPLORE_EFFICIENCY_KEY, EXPLOIT_EFFICIENCY_KEY,
                                      MEDIAN_EXPLOIT_LENGTH_KEY, AVERAGE_SPEED_KEY, FRACTION_GALLERY_IN_EXPLORE_KEY,
                                      FRACTION_TIME_IN_EXPLORE_KEY, EFFICIENCY_RATIO_KEY, EXPLORE_SPEED_KEY,
                                      EXPLOIT_SPEED_KEY, DEFAULT_FINAL_OUTPUT_FILENAME, EXCLUSION_REASON_KEY,
                                      STEP_ORIG_KEY, FRACTION_STEPS_UNIQUELY_COVERED_KEY, GALLERY_ORIG_KEY,
                                      GALLERY_ORIG_EXPLORE_KEY, GALLERY_ORIG_EXPLOIT_KEY,
                                      FRACTION_GALLERIES_UNIQUELY_COVERED_KEY, FRACTION_CLUSTERS_IN_GC_KEY,
                                      FRACTION_GALLERIES_UNIQUELY_COVERED_EXPLORE_KEY,
                                      FRACTION_GALLERIES_UNIQUELY_COVERED_EXPLOIT_KEY, N_CLUSTERS_IN_GC_KEY,
                                      ABSOLUTE_FEATURES_MESSAGE, RELATIVE_FEATURES_MESSAGE, EXPLORE_OUTLIER_REASON,
                                      EXPLOIT_OUTLIER_REASON, NO_EXPLOIT_EXCLUSION_REASON, MANUAL_EXCLUSION_REASON,
                                      GAME_LENGTH_EXCLUSION_REASON, GAME_DURATION_EXCLUSION_REASON, NO_GALLERY_REASON,
                                      PAUSE_EXCLUSION_REASON, SAMPLE_RELATIVE_FEATURES_LABEL,
                                      ROBUST_MEDIAN_PACE_KEY, ROBUST_THRESHOLD_KEY)
from CFGpy.behavioral import Configuration
from CFGpy.behavioral._utils import load_json, is_semantic_connection
from functools import reduce
from scipy.stats import zscore
from CFGpy.utils import get_vanilla_stats, step_orig_map_factory, gallery_orig_map_factory
from tqdm import tqdm


def _get_frac_uniquely_covered(player_objects, objects_not_uniquely_covered):
    set_player_objects = set(player_objects)
    n_unique_player_objects = len(set_player_objects)
    if not n_unique_player_objects:
        return None

    n_not_uniquely_covered = len(set_player_objects & set(objects_not_uniquely_covered))
    frac_not_uniquely_covered = n_not_uniquely_covered / n_unique_player_objects
    frac_uniquely_covered = 1 - frac_not_uniquely_covered
    return frac_uniquely_covered


class FeatureExtractor:
    def __init__(self, *, preprocessed_data,  is_rm1: bool = False,  is_mri: bool = False,
                 config: Configuration = None):
        self.config = config if config is not None else Configuration.default(is_rm1=is_rm1, is_mri=is_mri)
        self.input_data = PostparsedDataset(input_data=preprocessed_data, config=config)
        self.all_absolute_features = None
        self.output_df = None
        self.exclusions = pd.DataFrame(columns=[FEATURES_ID_KEY, EXCLUSION_REASON_KEY])
    
    @classmethod
    def from_json(cls, path: str, config=Configuration.default()):
        return cls(preprocessed_data=load_json(path), config=config)

    def extract(self, verbose=False):
        # 1. Absolute Features
        self.all_absolute_features = self._extract_absolute_features(verbose)
        self.output_df = self.all_absolute_features.copy()

        # --- ADD THIS LINE HERE ---
        self.output_df['is_excluded'] = False

        self._drop_nonfirst_games()

        # 2. Sanitize (Remove participants with 100% explore time)
        self._sanitize_features_for_analysis()

        # 3. Vanilla Relative Features
        # Now this will run because 'is_excluded' exists (even if all are False)
        vanilla_relative_features = self._extract_relative_features(get_vanilla_stats(), verbose=verbose)
        self.output_df = self.output_df.merge(vanilla_relative_features, on=FEATURES_ID_KEY)

        # 4. Soft Filters (This actually populates the real exclusion flags)
        self._apply_soft_filters()

        # 5. Sample-based Relative Features
        sample_relative_features = self._extract_relative_features(
            self.input_data.get_stats(),
            verbose=verbose,
            label=SAMPLE_RELATIVE_FEATURES_LABEL
        )
        self.output_df = self.output_df.merge(sample_relative_features, on=FEATURES_ID_KEY, how="left")

        return self.output_df

    def dump(self, path=DEFAULT_FINAL_OUTPUT_FILENAME):
        self.output_df.to_csv(path, index=False)  # reorder columns
        self.exclusions.to_csv(f"{path}_exclusions.csv", index=False)
        self.config.to_yaml(path)

        # TODO: document all filtered ids and filtering criteria
        # TODO: write html with dashboards to inspect data quality and some summary stats

    def is_cluster_in_GC(self, cluster, GC):
        for GC_cluster in GC:
            if is_semantic_connection(cluster, GC_cluster, self.config.MIN_OVERLAP_FOR_SEMANTIC_CONNECTION):
                return True

        return False

    def get_all_absolute_features(self):
        return self.all_absolute_features

    def _drop_nonfirst_games(self):
        """
        Selection logic:
        - If multiple games: if game 1 < 90s, keep game 2.
        - Otherwise: keep game 1.
        """
        # Ensure we are working with sorted indices
        all_games = self.all_absolute_features.copy()
        all_games['original_index'] = all_games.index
        all_games = all_games.sort_values(by=[FEATURES_ID_KEY, FEATURES_START_TIME_KEY])

        final_indices_to_keep = []

        for pid, group in all_games.groupby(FEATURES_ID_KEY):
            if len(group) == 1:
                final_indices_to_keep.append(group['original_index'].iloc[0])
            else:
                first_game = group.iloc[0]
                if first_game[GAME_DURATION_KEY] < 90:
                    # Log the false start exclusion
                    new_exclusion = pd.DataFrame({
                        FEATURES_ID_KEY: [pid],
                        EXCLUSION_REASON_KEY: [f"False start (<90s, kept game 2 of {len(group)})"]
                    })
                    self.exclusions = pd.concat([self.exclusions, new_exclusion], ignore_index=True)
                    # Use the second game
                    final_indices_to_keep.append(group['original_index'].iloc[1])
                else:
                    # First game was valid
                    final_indices_to_keep.append(group['original_index'].iloc[0])

        # Filter the DataFrame
        # We sort by original_index to maintain the order the dataset wrapper expects
        final_indices_to_keep.sort()
        self.output_df = self.all_absolute_features.loc[final_indices_to_keep].reset_index(drop=True)

        # Sync the Subject data objects!
        self.input_data.keep_only_indices(final_indices_to_keep)

    def _sanitize_features_for_analysis(self):
        """
        Removes participants who cannot be analyzed (no exploit phase)
        before any group statistics (Z-scores) are calculated.
        """
        # Identify participants with no exploit phases
        to_exclude = (
                (self.output_df[N_CLUSTERS_KEY] == 0) |
                (self.output_df[MEDIAN_EXPLOIT_LENGTH_KEY].isna()) |
                (self.output_df[FRACTION_TIME_IN_EXPLORE_KEY] >= 1.0)
        )

        # Log them in the exclusions dataframe so we don't lose the record
        no_exploit_ids = self.output_df.loc[to_exclude, FEATURES_ID_KEY]
        for pid in no_exploit_ids:
            new_exclusion = pd.DataFrame({
                FEATURES_ID_KEY: [pid],
                EXCLUSION_REASON_KEY: [NO_EXPLOIT_EXCLUSION_REASON]
            })
            self.exclusions = pd.concat([self.exclusions, new_exclusion], ignore_index=True)

        # CRITICAL: Drop them from the output_df before Z-scoring
        self.output_df = self.output_df[~to_exclude].reset_index(drop=True)

        # Now fill the specific MRI NaNs with 0 for the remaining valid participants
        mri_cols = [FRACTION_CLUSTERS_IN_GC_KEY, GALLERY_ORIG_EXPLOIT_KEY, GALLERY_ORIG_EXPLORE_KEY]
        for col in mri_cols:
            if col in self.output_df.columns:
                self.output_df[col] = self.output_df[col].fillna(0)

    def _apply_soft_filters(self):
        """Applies filters to the data and normalizes IDs to prevent IndexingErrors."""
        # 1. Initialize masks
        is_excluded = pd.Series(False, index=self.output_df.index)
        all_reasons = pd.Series("", index=self.output_df.index)

        # 2. Hard-coded call to your specific filter getters (as defined in your class)
        # We process absolute filters first, then relative filters
        for filter_func in [self._get_absolute_filters, self._get_sample_relative_filters]:
            masks, reasons = filter_func()
            for mask, reason in zip(masks, reasons):
                is_excluded |= mask
                all_reasons = all_reasons.where(~mask, all_reasons + reason + "; ")

        # --- THE ID ALIGNMENT FIX ---
        # Normalize the internal data objects so '092' becomes '92'
        # to match the Series index of is_excluded.
        try:
            for player_game in self.input_data:
                if hasattr(player_game, 'playerId'):
                    player_game.playerId = str(player_game.playerId).lstrip('0') or '0'
                elif hasattr(player_game, 'id'):
                    player_game.id = str(player_game.id).lstrip('0') or '0'
                elif isinstance(player_game, dict):
                    for key in ['playerId', 'id', 'player_id']:
                        if key in player_game:
                            player_game[key] = str(player_game[key]).lstrip('0') or '0'
        except TypeError:
            pass

        # 3. Apply the filter using the aligned boolean Series
        try:
            self.input_data.filter(~is_excluded)
        except Exception as e:
            print(f"Soft filter alignment warning: {e}")

        # 4. Update the output dataframe
        # 'is_excluded' is a standard flag, EXCLUSION_REASON_KEY is 'reason' in your consts.
        self.output_df['is_excluded'] = is_excluded
        self.output_df[EXCLUSION_REASON_KEY] = all_reasons.str.strip("; ")

    def _get_absolute_filters(self):
        """
        Absolute filters catch data integrity issues before sample-wide
        statistics (like Z-scores) are calculated.
        """
        reasons = (MANUAL_EXCLUSION_REASON, NO_EXPLOIT_EXCLUSION_REASON,
                   GAME_LENGTH_EXCLUSION_REASON, GAME_DURATION_EXCLUSION_REASON,
                   PAUSE_EXCLUSION_REASON, NO_GALLERY_REASON)

        # Hard-flag subjects with no exploit data
        # If median scav steps is NaN, they MUST be excluded here.
        no_exploit_mask = (
                (self.output_df[N_CLUSTERS_KEY] == 0) |
                (self.output_df[MEDIAN_EXPLOIT_LENGTH_KEY].isna()) |
                (self.output_df[FRACTION_TIME_IN_EXPLORE_KEY] >= 1.0)
        )

        no_gallery_mask = (self.output_df[N_GALLERIES_KEY] == 0)

        masks = (
            self.output_df[FEATURES_ID_KEY].isin(self.config.MANUALLY_EXCLUDED_IDS),
            no_exploit_mask,
            self.output_df[N_MOVES_KEY] < self.config.MIN_N_MOVES,
            self.output_df[GAME_DURATION_KEY] < self.config.MIN_GAME_DURATION_SEC,
            self.output_df[LONGEST_PAUSE_KEY] > self.config.MAX_PAUSE_DURATION_SEC,
            no_gallery_mask
        )

        return masks, reasons

    def _get_sample_relative_filters(self):
        reasons = (EXPLORE_OUTLIER_REASON, EXPLOIT_OUTLIER_REASON)

        # We only calculate Z-scores for participants who haven't
        # already been excluded by the absolute filters (like 'No Exploit').
        # This prevents the "NaN column" crash.
        valid_mask = ~self.output_df['is_excluded']

        if not valid_mask.any():
            raise ValueError("All subjects were excluded before relative filtering. "
                             "Check if exploitation detection is working.")

        target_cols = [MEDIAN_EXPLORE_LENGTH_KEY, MEDIAN_EXPLOIT_LENGTH_KEY]

        # Initialize z-scores with NaNs
        zscore_results = pd.DataFrame(np.nan, index=self.output_df.index, columns=target_cols)

        # Calculate Z-scores ONLY for the valid subset
        subset = self.output_df.loc[valid_mask, target_cols]
        zscore_results.loc[valid_mask] = subset.apply(zscore)

        masks = (abs(zscore_results[MEDIAN_EXPLORE_LENGTH_KEY]) > self.config.MAX_ZSCORE_FOR_OUTLIERS,
                 abs(zscore_results[MEDIAN_EXPLOIT_LENGTH_KEY]) > self.config.MAX_ZSCORE_FOR_OUTLIERS)

        return masks, reasons

    def _update_exclusion_info(self, masks, reasons):
        """
        Updates self.to_exclude based on filters results.
        :param masks: a collection of masks, each has **True for players to exclude**, false for players to keep.
        :param reasons: a collection of strings describing exclusion reasons for the masks.
        """
        for is_excluded, reason in zip(masks, reasons):
            ids_to_exclude = self.output_df.loc[is_excluded, FEATURES_ID_KEY]
            current_exclusion = pd.DataFrame({
                FEATURES_ID_KEY: ids_to_exclude,
                EXCLUSION_REASON_KEY: [reason] * len(ids_to_exclude)
            })
            self.exclusions = pd.concat((self.exclusions, current_exclusion))

    def _extract_absolute_features(self, verbose=True):
        n_galleries_in_explore = []
        total_explore_times = []
        total_exploit_times = []
        total_explore_lengths = []
        total_exploit_lengths = []

        iterator = self.input_data
        if verbose:
            print(ABSOLUTE_FEATURES_MESSAGE)
            iterator = tqdm(iterator)

        absolute_features = []
        for player_data in iterator:
            # pre-calculations
            explore_lengths = [end - start for start, end in player_data.explore_slices]
            exploit_lengths = [end - start for start, end in player_data.exploit_slices]
            is_gallery = player_data.get_gallery_mask()

            # TODO: Added from the aviv repo, to handle edge case of no galleries
            n_galleries = sum(is_gallery)
            if n_galleries == 0:
                print(f"Player {player_data.id} has no galleries, skipping...")
                continue
            is_explore = player_data.get_explore_mask()

            # data collection for later vectorized operations
            n_galleries_in_explore.append(sum(is_gallery & is_explore))
            total_explore_times.append(player_data.total_explore_time())
            total_exploit_times.append(player_data.total_exploit_time())
            total_explore_lengths.append(sum(explore_lengths))
            total_exploit_lengths.append(sum(exploit_lengths))
            # the values that are calculated only in MRI mode
            # TODO: added, didn't exist in the original MeasureCalculator in "aviv" repo
            robust_median = getattr(player_data, ROBUST_MEDIAN_PACE_KEY, np.nan)
            robust_threshold = getattr(player_data, ROBUST_THRESHOLD_KEY, np.nan)

            # player-wise calculations
            explore_efficiency, exploit_efficiency = player_data.get_efficiency()
            absolute_features.append({
                FEATURES_ID_KEY: player_data.id,
                FEATURES_START_TIME_KEY: datetime.fromtimestamp(player_data.start_time).isoformat(),
                GAME_DURATION_KEY: player_data.get_last_action_time(),
                N_MOVES_KEY: len(player_data),
                N_GALLERIES_KEY: n_galleries,
                SELF_AVOIDANCE_KEY: player_data.get_self_avoidance(),
                N_CLUSTERS_KEY: len(player_data.exploit_slices),
                EXPLORE_EFFICIENCY_KEY: explore_efficiency,
                EXPLOIT_EFFICIENCY_KEY: exploit_efficiency,
                MEDIAN_EXPLORE_LENGTH_KEY: np.median(explore_lengths),
                MEDIAN_EXPLOIT_LENGTH_KEY: np.median(exploit_lengths),
                LONGEST_PAUSE_KEY: player_data.get_max_pause_duration(),
                ROBUST_MEDIAN_PACE_KEY: robust_median,
                ROBUST_THRESHOLD_KEY: robust_threshold,
            })

        # vectorized operations
        features_df = pd.DataFrame(absolute_features)
        features_df[AVERAGE_SPEED_KEY] = features_df[N_MOVES_KEY] / features_df[GAME_DURATION_KEY]
        features_df[FRACTION_GALLERY_IN_EXPLORE_KEY] = pd.Series(n_galleries_in_explore) / features_df[N_GALLERIES_KEY]
        features_df[FRACTION_TIME_IN_EXPLORE_KEY] = pd.Series(total_explore_times) / features_df[GAME_DURATION_KEY]
        features_df[EFFICIENCY_RATIO_KEY] = features_df[EXPLORE_EFFICIENCY_KEY] / features_df[EXPLOIT_EFFICIENCY_KEY]
        features_df[EXPLORE_SPEED_KEY] = pd.Series(total_explore_lengths) / pd.Series(total_explore_times)
        features_df[EXPLOIT_SPEED_KEY] = pd.Series(total_exploit_lengths) / pd.Series(total_exploit_times)

        # Ensure percentages and origins are 0 if no clusters/galleries exist
        cols_to_zero = [FRACTION_GALLERY_IN_EXPLORE_KEY, FRACTION_TIME_IN_EXPLORE_KEY]
        for col in cols_to_zero:
            if col in features_df.columns:
                features_df[col] = features_df[col].fillna(0)

        return features_df

    def _extract_relative_features(self, stats, label=None, verbose=False):
        steps_not_uniquely_covered, step_counter, galleries_not_uniquely_covered, gallery_counter, GC = stats
        label_ext = f" ({label})" if label else ""

        step_orig_map = step_orig_map_factory(step_counter, alpha=self.config.STEP_ORIG_PSEUDOCOUNT,
                                              d=self.config.STEP_ORIG_N_CATEGORIES)
        gallery_orig_map = gallery_orig_map_factory(gallery_counter, alpha=self.config.GALLERY_ORIG_PSEUDOCOUNT,
                                                    d=self.config.GALLERY_ORIG_N_CATEGORIES)

        iterator = self.input_data
        if verbose:
            print(RELATIVE_FEATURES_MESSAGE.format(label_ext))
            iterator = tqdm(iterator)

        relative_features = []
        for player_data in iterator:
            steps = player_data.get_steps()
            step_orig = [step_orig_map[step] for step in steps]
            gallery_ids = player_data.get_gallery_ids()
            gallery_orig = np.array([gallery_orig_map[shape_id] for shape_id in gallery_ids])
            is_gallery = player_data.get_gallery_mask()
            is_explore_given_gallery = player_data.get_explore_mask()[is_gallery]
            is_exploit_given_gallery = ~is_explore_given_gallery

            exploit_clusters = player_data.get_exploit_clusters()
            n_clusters_in_GC = sum([self.is_cluster_in_GC(cluster, GC) for cluster in exploit_clusters])

            # Use 0 if there are no exploit slices to avoid NaN here
            frac_clusters_in_GC = (n_clusters_in_GC / len(player_data.exploit_slices)
                                   if player_data.exploit_slices else 0)

            # SAFETY CHECK: If there is no exploit gallery, np.mean() returns NaN
            # We handle them explicitly here so the loop is "quiet"
            gall_orig_exploit = np.mean(gallery_orig[is_exploit_given_gallery]) if any(
                is_exploit_given_gallery) else 0
            gall_orig_explore = np.mean(gallery_orig[is_explore_given_gallery]) if any(
                is_explore_given_gallery) else 0

            relative_features.append({
                FEATURES_ID_KEY: player_data.id,
                f"{STEP_ORIG_KEY}{label_ext}": np.mean(step_orig),
                f"{FRACTION_STEPS_UNIQUELY_COVERED_KEY}{label_ext}":
                    _get_frac_uniquely_covered(steps, steps_not_uniquely_covered),
                f"{GALLERY_ORIG_KEY}{label_ext}": np.mean(gallery_orig) if len(gallery_orig) > 0 else 0,
                f"{GALLERY_ORIG_EXPLORE_KEY}{label_ext}": gall_orig_explore,
                f"{GALLERY_ORIG_EXPLOIT_KEY}{label_ext}": gall_orig_exploit,
                f"{FRACTION_GALLERIES_UNIQUELY_COVERED_KEY}{label_ext}":
                    _get_frac_uniquely_covered(gallery_ids, galleries_not_uniquely_covered),
                f"{FRACTION_GALLERIES_UNIQUELY_COVERED_EXPLORE_KEY}{label_ext}":
                    _get_frac_uniquely_covered(gallery_ids[is_explore_given_gallery],
                                               galleries_not_uniquely_covered),
                f"{FRACTION_GALLERIES_UNIQUELY_COVERED_EXPLOIT_KEY}{label_ext}":
                    _get_frac_uniquely_covered(gallery_ids[is_exploit_given_gallery],
                                               galleries_not_uniquely_covered),
                f"{N_CLUSTERS_IN_GC_KEY}{label_ext}": n_clusters_in_GC,
                f"{FRACTION_CLUSTERS_IN_GC_KEY}{label_ext}": frac_clusters_in_GC,
            })

        rel_df = pd.DataFrame(relative_features)

        # Final cleanup for the DataFrame
        mri_zero_fill = [
            f"{FRACTION_CLUSTERS_IN_GC_KEY}{label_ext}",
            f"{FRACTION_GALLERIES_UNIQUELY_COVERED_EXPLOIT_KEY}{label_ext}",
            f"{FRACTION_GALLERIES_UNIQUELY_COVERED_EXPLORE_KEY}{label_ext}",
            f"{FRACTION_GALLERIES_UNIQUELY_COVERED_KEY}{label_ext}",
            f"{GALLERY_ORIG_EXPLOIT_KEY}{label_ext}",
            f"{GALLERY_ORIG_EXPLORE_KEY}{label_ext}",
            f"{GALLERY_ORIG_KEY}{label_ext}"
        ]

        for col in mri_zero_fill:
            if col in rel_df.columns:
                rel_df[col] = rel_df[col].fillna(0)

        return rel_df
