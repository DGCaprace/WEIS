from ctypes import (
	CDLL,
    POINTER,
    create_string_buffer,
    byref,
    c_int,
    c_double,
    c_char,
    c_bool
)
import os
from typing import List
import numpy as np


class FastLibAPI(CDLL):

    def __init__(self, library_path: str, input_file_name: str):
        super().__init__(library_path)
        self.library_path = library_path
        self.input_file_name = create_string_buffer(os.path.abspath(input_file_name).encode('utf-8'))

        self._initialize_routines()

        # Create buffers for class data
        self.n_turbines = c_int(1)
        self.i_turb = c_int(0)
        self.dt = c_double(0.0)
        self.t_max = c_double(0.0)
        self.abort_error_level = c_int(99)
        self.end_early = c_bool(False)
        self.num_outs = c_int(0)
        self._channel_names = create_string_buffer(20 * 4000)
        self.output_array = None

        # The inputs are meant to be from Simulink.
        # If < 8, FAST_SetExternalInputs simply returns,
        # but this behavior may change to an error
        ### MAKE THIS 8 OR 11
        self._num_inputs = c_int(51)
        self._inp_array = (c_double * 51)(0.0, )  # 10 is hard-coded in FAST_Library as MAXInitINPUTS
        self._inp_array[0] = -1.0  # Sensor type - 

        self.output_values = None
        self.ended = False


    def _initialize_routines(self):
        self.FAST_AllocateTurbines.argtypes = [
            POINTER(c_int),
            POINTER(c_int),
            POINTER(c_char)
        ]
        self.FAST_AllocateTurbines.restype = c_int

        self.FAST_Sizes.argtype = [
            POINTER(c_int),         # iTurb IN
            POINTER(c_char),        # InputFileName_c IN
            POINTER(c_int),         # AbortErrLev_c OUT
            POINTER(c_int),         # NumOuts_c OUT
            POINTER(c_double),      # dt_c OUT
            POINTER(c_double),      # tmax_c OUT
            POINTER(c_int),         # ErrStat_c OUT
            POINTER(c_char),        # ErrMsg_c OUT
            POINTER(c_char),        # ChannelNames_c OUT
            POINTER(c_double),      # TMax OPTIONAL IN
            POINTER(c_double)       # InitInpAry OPTIONAL IN
        ]
        self.FAST_Sizes.restype = c_int

        self.FAST_Start.argtype = [
            POINTER(c_int),         # iTurb IN
            POINTER(c_int),         # NumInputs_c IN
            POINTER(c_int),         # NumOutputs_c IN
            POINTER(c_double),      # InputAry IN
            POINTER(c_double),      # OutputAry OUT
            POINTER(c_int),         # ErrStat_c OUT
            POINTER(c_char)         # ErrMsg_c OUT
        ]
        self.FAST_Start.restype = c_int

        self.FAST_Update.argtype = [
            POINTER(c_int),         # iTurb IN
            POINTER(c_int),         # NumInputs_c IN
            POINTER(c_int),         # NumOutputs_c IN
            POINTER(c_double),      # InputAry IN
            POINTER(c_double),      # OutputAry OUT
            POINTER(c_bool),        # EndSimulationEarly OUT
            POINTER(c_int),         # ErrStat_c OUT
            POINTER(c_char)         # ErrMsg_c OUT
        ]
        self.FAST_Update.restype = c_int

        self.FAST_DeallocateTurbines.argtypes = [
            POINTER(c_int),         # ErrStat_c OUT
            POINTER(c_char),        # ErrMsg_c OUT
        ]
        self.FAST_DeallocateTurbines.restype = c_int

        self.FAST_End.argtypes = [
            POINTER(c_int),         # iTurb IN
            POINTER(c_bool),        # StopTheProgram IN
        ]
        self.FAST_End.restype = c_int


    def fatal_error(self, error_status) -> bool:
        return error_status.value >= self.abort_error_level.value


    def fast_init(self):
        _error_status = c_int(0)
        _error_message = create_string_buffer(1025)

        self.FAST_AllocateTurbines(
            byref(self.n_turbines),
            byref(_error_status),
            _error_message
        )
        if self.fatal_error(_error_status):
            raise RuntimeError(f"Error {_error_status.value}: {_error_message.value}")

        self.FAST_Sizes(
            byref(self.i_turb),
            self.input_file_name,
            byref(self.abort_error_level),
            byref(self.num_outs),
            byref(self.dt),
            byref(self.t_max),
            byref(_error_status),
            _error_message,
            self._channel_names,
            None,   # Optional arguments must pass C-Null pointer; with ctypes, use None.
            None    # Optional arguments must pass C-Null pointer; with ctypes, use None.
        )
        if self.fatal_error(_error_status):
            raise RuntimeError(f"Error {_error_status.value}: {_error_message.value}")

        # Allocate the data for the outputs
        # NOTE: The ctypes array allocation (output_array) must be after the output_values
        # allocation, or otherwise seg fault.
        self.output_values = np.empty( (self.total_time_steps, self.num_outs.value) )
        self.output_array = (c_double * self.num_outs.value)(0.0, )


    def fast_sim(self):
        _error_status = c_int(0)
        _error_message = create_string_buffer(1025)

        self.FAST_Start(
            byref(self.i_turb),
            byref(self._num_inputs),
            byref(self.num_outs),
            byref(self._inp_array),
            byref(self.output_array),
            byref(_error_status),
            _error_message
        )
        self.output_values[0] = self.output_array[:]
        if self.fatal_error(_error_status):
            self.fast_deinit()
            raise RuntimeError(f"Error {_error_status.value}: {_error_message.value}")

        for i in range( 1, self.total_time_steps ):
            self.FAST_Update(
                byref(self.i_turb),
                byref(self._num_inputs),
                byref(self.num_outs),
                byref(self._inp_array),
                byref(self.output_array),
                byref(self.end_early),
                byref(_error_status),
                _error_message
            )
            self.output_values[i] = self.output_array[:]
            if self.fatal_error(_error_status):
                self.fast_deinit()
                raise RuntimeError(f"Error {_error_status.value}: {_error_message.value}")
            if self.end_early:
                break


    def fast_deinit(self):
        _error_status = c_int(0)
        _error_message = create_string_buffer(1025)

        if not self.ended:
            self.ended = True

            # Deallocate all the internal variables and allocatable arrays
            # Despite the name, this does not actually end the program
            self.FAST_End(
                byref(self.i_turb),
                byref(c_bool(False))
            )

            # Deallocate the Turbine array
            self.FAST_DeallocateTurbines(
                byref(_error_status),
                _error_message
            )
            if self.fatal_error(_error_status):
                raise RuntimeError(f"Error {_error_status.value}: {_error_message.value}")


    def fast_run(self):
        self.fast_init()
        self.fast_sim()
        self.fast_deinit()


    @property
    def total_time_steps(self):
        return int(self.t_max.value / self.dt.value) + 1


    @property
    def output_channel_names(self) -> List:
        if len(self._channel_names.value.split()) == 0:
            return []
        output_channel_names = self._channel_names.value.split()
        output_channel_names = [n.decode('UTF-8') for n in output_channel_names]        
        return output_channel_names
