from os import PathLike
import os
import numpy as np
import pandas as pd
import warnings

from flowio import FlowData
from flowio.exceptions import FCSParsingError

from typing import Optional, Union


class FCSFile:
    """
    intermediate representation of a sample FCS file.
    Organization into an object is meant to facilitate cleaner code
    """

    def __init__(self,
                 input_directory: Union[PathLike, str],
                 file_name: str,
                 subsample: Optional[int] = None,
                 truncate_max_range: bool = True
                 ) -> None:

        self.original_filename = file_name

        raw_data = self._load_fcs_file_from_disk(input_directory,
                                                 file_name,
                                                 ignore_offset_error = False)

        self.compensation_status = "uncompensated"
        self.transform_status = "untransformed"
        self.gating_status = "ungated"

        self._fcs_event_count = self._parse_event_count(raw_data)
        self.version = self._parse_fcs_version(raw_data)
        self.fcs_metadata = self._parse_fcs_metadata(raw_data)
        self.channels = self._parse_channel_information(raw_data)
        self.original_events = \
            self._parse_and_process_original_events(raw_data,
                                                    subsample,
                                                    truncate_max_range)
        self.event_count = self.original_events.shape[0]

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'v{self.version}, '
            f'{self.original_filename}, '
            f'{self.channels.shape[0]} channels, '
            f'{self.event_count} events, '
            f'gating status: {self.gating_status}, '
            f'compensation status: {self.compensation_status}, '
            f'transform status: {self.transform_status})'
        )

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            data = self.original_events,
            index = pd.Index(list(range(self.event_count))),
            columns = self.channels.index
        )

    def get_events(self,
                   source: str = "raw") -> Optional[np.ndarray]:
        """returns the events"""
        if source == "raw":
            return self._get_original_events()
        else:
            raise NotImplementedError(
                "Only Raw ('raw') events can be fetched."
            )

    def _get_original_events(self) -> np.ndarray:
        """returns uncompensated original events"""
        return self.original_events

    def get_channel_index(self,
                          channel_label: str) -> int:
        """
        performs a lookup in the channels dataframe and
        returns the channel index by the fcs file channel numbers
        """
        return self.channels.loc[
            self.channels.index == channel_label,
            "channel_numbers"
        ].iloc[0] - 1

    def _parse_event_count(self,
                           fcs_data: FlowData) -> int:
        """returns the total event count"""
        return fcs_data.event_count

    def _subsample_events(self,
                          events: np.ndarray,
                          size: int) -> np.ndarray:
        """subsamples the data array using a user defined number of cells"""
        if size >= events.shape[0]:
            return events

        return events[np.random.randint(events.shape[0],
                                        size = size), :]

    def _parse_and_process_original_events(self,
                                           fcs_data: FlowData,
                                           subsample: Optional[int],
                                           truncate_max_range: bool) -> np.ndarray:  # noqa
        """parses and processes the original events"""
        tmp_orig_events = self._parse_original_events(fcs_data)
        if subsample is not None:
            tmp_orig_events = self._subsample_events(tmp_orig_events,
                                                     subsample)
        tmp_orig_events = self._process_original_events(tmp_orig_events,
                                                        truncate_max_range)
        return tmp_orig_events

    def _process_original_events(self,
                                 tmp_orig_events: np.ndarray,
                                 truncate_max_range: bool) -> np.ndarray:
        """
        processes the original events by convolving the channel gains
        the decades and the time channel
        """
        if truncate_max_range:
            tmp_orig_events = self._adjust_range(tmp_orig_events)
        tmp_orig_events = self._remove_nans_from_events(tmp_orig_events)
        tmp_orig_events = self._adjust_time_channel(tmp_orig_events)
        tmp_orig_events = self._adjust_decades(tmp_orig_events)
        tmp_orig_events = self._adjust_channel_gain(tmp_orig_events)
        return tmp_orig_events

    def _adjust_range(self,
                      arr: np.ndarray) -> np.ndarray:
        channel_ranges = self.channels["pnr"].to_numpy()
        range_exceeded_cells = (arr > channel_ranges)
        range_exceeded_channels = range_exceeded_cells.any(axis = 0)
        if any(range_exceeded_channels):
            exceeded_channels = self.channels[range_exceeded_channels].index.tolist()
            number_of_exceeded_cells = range_exceeded_cells.sum(axis = 0)
            TruncationWarning(exceeded_channels, number_of_exceeded_cells)
            array_mins = np.min(arr, axis = 0)
            return np.clip(arr, array_mins, channel_ranges)
        return arr

    def _remove_nans_from_events(self,
                                 arr: np.ndarray) -> np.ndarray:
        """Function to remove rows with NaN, inf and -inf"""
        if np.isinf(arr).any():
            idxs = np.argwhere(np.isinf(arr))[:, 0]
            arr = arr[~np.in1d(np.arange(arr.shape[0]), idxs)]
            warning_message = (
                f"{idxs.shape[0]} cells were removed from "
                f"{self.original_filename} due to "
                "the presence of Infinity values"
            )
            InfRemovalWarning(warning_message)
        if np.isnan(arr).any():
            idxs = np.argwhere(np.isnan(arr))[:, 0]
            arr = arr[~np.in1d(np.arange(arr.shape[0]), idxs)]
            warning_message = (
                f"{idxs.shape[0]} cells were removed from "
                f"{self.original_filename} due to "
                "the presence of NaN values"
            )
            NaNRemovalWarning(warning_message)
        return arr

    def _adjust_channel_gain(self,
                             events: np.ndarray) -> np.ndarray:
        """divides the event fluorescence values by the channel gain"""
        channel_gains = self.channels.sort_values("channel_numbers")["png"].to_numpy()  # noqa
        return np.divide(events, channel_gains)

    def _adjust_decades(self,
                        events: np.ndarray) -> np.ndarray:
        """adjusts the decades"""
        for (decades, log0), \
                channel_number, \
                channel_range in zip(self.channels["pne"],
                                     self.channels["channel_numbers"],
                                     self.channels["pnr"]):
            if decades > 0:
                events[:, channel_number - 1] = (
                    10 ** (decades * events[:, channel_number - 1] / channel_range)  # noqa
                ) * log0

        return events

    def _adjust_time_channel(self,
                             events: np.ndarray) -> np.ndarray:
        """multiplies the time values by the time step"""
        if self._time_channel_exists:
            time_index, time_step = self._find_time_channel()
            events[:, time_index] = events[:, time_index] * time_step
        return events

    def _find_time_channel(self) -> tuple[int, float]:
        """returns the index and time_step of the time channel if present"""
        if "timestep" in self.fcs_metadata:
            time_step = float(self.fcs_metadata["timestep"])
        else:
            time_step = 1.0
        time_index = int(
            self.channels.loc[
                self.channels.index.isin(["Time", "time"]), "channel_numbers"
            ].iloc[0]
        ) - 1
        return (time_index, time_step)

    def _time_channel_exists(self) -> bool:
        """returns bool if time channel exists"""
        return any(
            time_symbol in self.channels.index
            for time_symbol in ["Time", "time"]
        )

    def _parse_original_events(self,
                               fcs_data: FlowData) -> np.ndarray:
        """function to parse the original events from the fcs file"""
        return np.array(
            fcs_data.events,
            dtype=np.float64,
            order = "C"
        ).reshape(-1, fcs_data.channel_count)

    def _remove_disallowed_characters_from_string(self,
                                                  input_string: str) -> str:
        """ function to remove disallowed characters from the string"""
        for char in [" ", "/", "-"]:
            if char in input_string:
                input_string = input_string.replace(char, "_")
        return input_string

    def _parse_channel_information(self,
                                   fcs_data: FlowData) -> pd.DataFrame:
        """\
        retrieves the channel information from the
        fcs file and returns a dataframe
        """
        channels: dict = fcs_data.channels
        pnn_labels = [self._parse_pnn_label(channels, channel_number) for
                      channel_number in channels]
        pns_labels = [self._parse_pns_label(channels, channel_number) for
                      channel_number in channels]
        channel_gains = [self._parse_channel_gain(channel_number) for
                         channel_number in channels]
        channel_lin_log = [self._parse_channel_lin_log(channel_number) for
                           channel_number in channels]
        channel_ranges = [self._parse_channel_range(channel_number) for
                          channel_number in channels]

        channel_numbers = [int(k) for k in channels]

        channel_frame = pd.DataFrame(
            data = {"pns": pns_labels,
                    "png": channel_gains,
                    "pne": channel_lin_log,
                    "pnr": channel_ranges,
                    "channel_numbers": channel_numbers
                    },
            index = pnn_labels
        )

        return channel_frame.sort_values("channel_numbers")

    def _parse_pnn_label(self,
                         channels: dict,
                         channel_number: str) -> str:
        """parses the pnn labels from the fcs file"""
        return channels[channel_number]["PnN"]

    def _parse_pns_label(self,
                         channels: dict,
                         channel_number: str) -> str:
        """parses the pns labels from the fcs file"""
        try:
            return self._remove_disallowed_characters_from_string(
                channels[channel_number]["PnS"]
            )
        except KeyError:
            return ""

    def _parse_channel_range(self,
                             channel_number: str) -> Union[int, float]:
        """parses the channel range from the fcs file"""
        try:
            return int(self.fcs_metadata[f"p{channel_number}r"])
        except ValueError as e:
            """
            Some FCS Files have deranged channel ranges which throw
            a conversion error. In order to avoid crashing, we return
            np.inf instead. These values are not needed down the road,
            so we do not issue a warning.
            """
            if "invalid literal for int() with base 10" in str(e):
                return np.inf
            else:
                raise ValueError from e

    def _parse_channel_lin_log(self,
                               channel_number: str) -> tuple[float, float]:
        """parses the channel lin log from the fcs file"""
        try:
            (decades, log0) = [
                float(x)
                for x in self.fcs_metadata[f"p{channel_number}e"].split(",")
            ]
            if log0 == 0.0 and decades != 0:
                log0 = 1.0  # FCS std states to use 1.0 for invalid 0 value
            return (decades, log0)
        except KeyError:
            return (0.0, 0.0)

    def _parse_channel_gain(self,
                            channel_number: str) -> float:
        """parses the channel gain from the fcs file"""
        if self.fcs_metadata[f"p{channel_number}n"] in ["Time", "time"]:
            return 1.0
        try:
            return float(self.fcs_metadata[f"p{channel_number}g"])
        except KeyError:
            return 1.0

    def _parse_fcs_metadata(self,
                            fcs_data: FlowData) -> dict:
        """Returns fcs metadata as a dictionary"""
        return fcs_data.text

    def _parse_fcs_version(self,
                           fcs_data: FlowData) -> Optional[str]:
        """returns the fcs version"""
        try:
            return str(fcs_data.header["version"])
        except KeyError:
            return None

    def _load_fcs_file_from_disk(self,
                                 input_directory: Union[PathLike, str],
                                 file_name: str,
                                 ignore_offset_error: bool) -> FlowData:
        """function to load the fcs from the hard rive"""
        try:
            return FlowData(
                os.path.join(input_directory, file_name),
                ignore_offset_error
            )
        except FCSParsingError:
            warnings.warn(
                "FACSPy IO: FCS file could not be read with "
                f"ignore_offset_error set to {ignore_offset_error}. "
                "Parameter is set to True."
            )
            return FlowData(
                os.path.join(input_directory, file_name),
                ignore_offset_error = True
            )


class NaNRemovalWarning(Warning):
    def __init__(self,
                 message) -> None:
        self.message = message
        warnings.warn(message, UserWarning)

    def __str__(self):
        return repr(self.message)


class TruncationWarning(Warning):
    def __init__(self,
                 exceeded_channels,
                 number_exceeded_cells) -> None:
        self.message = "Some data points exceed the PnR value. " + \
                       "The data points are truncated. To avoid " + \
                       "truncation, set the PnR value manually or " + \
                       "pass `truncate_max_range = False`. The " + \
                       "following counts were outside the channel range: "
        channel_count_mapping = [f"{ch}: {count}"
                                 for ch, count in
                                 zip(exceeded_channels, number_exceeded_cells)
                                 if count != 0]
        self.message += f"{', '.join(channel_count_mapping)}"
        warnings.warn(self.message, UserWarning)

    def __str__(self):
        return repr(self.message)


class InfRemovalWarning(Warning):
    def __init__(self,
                 message) -> None:
        self.message = message
        warnings.warn(message, UserWarning)

    def __str__(self):
        return repr(self.message)

