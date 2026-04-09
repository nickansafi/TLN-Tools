from f1tenth_benchmarks.classic_racing.RaceTrackGenerator import RaceTrackGenerator, load_parameter_file_with_extras
from f1tenth_benchmarks.classic_racing.GlobalPurePursuit import GlobalPurePursuit
from f1tenth_benchmarks.classic_racing.GlobalMPCC import GlobalMPCC
from f1tenth_benchmarks.mapless_racing.FollowTheGap import FollowTheGap
from f1tenth_benchmarks.drl_racing.EndToEndAgent import EndToEndAgent, TrainEndToEndAgent, TinyAgent, TrainTinyAgent
from f1tenth_benchmarks.zarrar.mlp_il import EndToEnd
from f1tenth_benchmarks.zarrar.tiny_lidarnet2 import TinyLidarNet

from f1tenth_benchmarks.data_tools.specific_plotting.plot_drl_training import plot_drl_training
from f1tenth_benchmarks.data_tools.plot_trajectory_analysis import plot_trajectory_analysis
from f1tenth_benchmarks.run_scripts.run_functions import *
import os

NUMBER_OF_LAPS = 10
PLOT = False

BENCH_ROOT = os.path.abspath(__file__).rpartition("/Benchmark")[0] + \
    '/Benchmark/f1tenth_benchmarks/zarrar/'

# Scale lists used during training — must mirror train2.py RUN_CONFIGS.
MULTIRES_SCALES = [1.0, 0.75]
SINGLE_100_SCALES = [1.0]
SINGLE_075_SCALES = [0.75]


def _tln_path(suffix: str) -> str:
    return os.path.join(BENCH_ROOT, f'TLN_{suffix}_noquantized.tflite')


def generate_racelines():
    map_list = ['example', 'MoscowRaceway']
    params = load_parameter_file_with_extras("RaceTrackGenerator", extra_params={"mu": 0.9})
    raceline_id = f"mu{int(params.mu*100)}"
    for map_name in map_list:
        RaceTrackGenerator(map_name, raceline_id, params, plot_raceline=True)


def optimisation_and_tracking():
    test_id = "benchmark_pp"
    planner = GlobalPurePursuit(test_id, False, planner_name="GlobalPlanPP",
                                extra_params={"racetrack_set": "mu90"})
    test_planning_all_maps(planner, test_id, number_of_laps=NUMBER_OF_LAPS)
    plot_trajectory_analysis(planner.name, test_id)


def mpcc():
    test_id = "benchmark_mpcc"
    planner = GlobalMPCC(test_id, False, planner_name="GlobalPlanMPCC",
                         extra_params={"friction_mu": 0.9})
    test_planning_all_maps(planner, test_id, number_of_laps=10)
    plot_trajectory_analysis(planner.name, test_id)


def follow_the_gap():
    test_id = "benchmark_ftg"
    planner = FollowTheGap(test_id)
    test_mapless_all_maps(planner, test_id, number_of_laps=NUMBER_OF_LAPS)
    plot_trajectory_analysis(planner.name, test_id)


def end_to_end_drl():
    test_id = "benchmark_e2e_drl"
    seed_randomness(12)
    testing_agent = EndToEndAgent(test_id)
    test_mapless_all_maps(testing_agent, test_id, number_of_laps=NUMBER_OF_LAPS)
    plot_trajectory_analysis(testing_agent.name, test_id)


def tinylidar_drl():
    test_id = "benchmark_tiny_drl"
    seed_randomness(12)
    testing_agent = TinyAgent(test_id)
    test_mapless_all_maps(testing_agent, test_id, number_of_laps=NUMBER_OF_LAPS)
    plot_trajectory_analysis(testing_agent.name, test_id)


# ----------------------------------------------------------
# Legacy TLN / MLP benchmarks (unchanged)
# ----------------------------------------------------------

def end_to_end_il():
    test_id = "benchmark_e2e_il"
    planner = EndToEnd(test_id, 4,
        os.path.abspath(__file__).rpartition("/Benchmark")[0] +
        '/Benchmark/f1tenth_benchmarks/zarrar/f1_tenth_model_diff_MLP_S_noquantized.tflite')
    test_mapless_all_maps(planner, test_id, number_of_laps=NUMBER_OF_LAPS)
    plot_trajectory_analysis(planner.name, test_id)


def end_to_end_il_m():
    test_id = "benchmark_e2e_il_m"
    planner = EndToEnd(test_id, 2,
        os.path.abspath(__file__).rpartition("/Benchmark")[0] +
        '/Benchmark/f1tenth_benchmarks/zarrar/f1_tenth_model_diff_MLP_M_noquantized.tflite')
    test_mapless_all_maps(planner, test_id, number_of_laps=NUMBER_OF_LAPS)
    plot_trajectory_analysis(planner.name, test_id)


def end_to_end_il_l():
    test_id = "benchmark_e2e_il_l"
    planner = EndToEnd(test_id, 1,
        os.path.abspath(__file__).rpartition("/Benchmark")[0] +
        '/Benchmark/f1tenth_benchmarks/zarrar/f1_tenth_model_diff_paper_noquantized.tflite')
    test_mapless_all_maps(planner, test_id, number_of_laps=NUMBER_OF_LAPS)
    plot_trajectory_analysis(planner.name, test_id)


def end_to_end_il_128():
    test_id = "benchmark_e2e_il_128"
    planner = EndToEnd(test_id, 1,
        os.path.abspath(__file__).rpartition("/Benchmark")[0] +
        '/Benchmark/f1tenth_benchmarks/zarrar/f1_tenth_model_diff_128_noquantized.tflite')
    test_mapless_all_maps(planner, test_id, number_of_laps=NUMBER_OF_LAPS)
    plot_trajectory_analysis(planner.name, test_id)


# ----------------------------------------------------------
# New multi-resolution TLN benchmarks
# ----------------------------------------------------------

def tinylidar_il_multires_shared(scale):
    """Multi-res model with shared affine, evaluated at the given scale."""
    test_id = f"benchmark_tiny_il_multires_sharedaffine_{scale}"
    print(test_id)
    planner = TinyLidarNet(
        test_id, 1, 0,
        _tln_path('multires_sharedaffine'),
        scale=scale,
        resolution_scales=MULTIRES_SCALES,
    )
    test_mapless_all_maps(planner, test_id, number_of_laps=NUMBER_OF_LAPS)
    if PLOT:
        plot_trajectory_analysis(planner.name, test_id)


def tinylidar_il_multires_perbank(scale):
    """Multi-res model with per-bank affine, evaluated at the given scale."""
    test_id = f"benchmark_tiny_il_multires_perbankaffine_{scale}"
    print(test_id)
    planner = TinyLidarNet(
        test_id, 1, 0,
        _tln_path('multires_perbankaffine'),
        scale=scale,
        resolution_scales=MULTIRES_SCALES,
    )
    test_mapless_all_maps(planner, test_id, number_of_laps=NUMBER_OF_LAPS)
    if PLOT:
        plot_trajectory_analysis(planner.name, test_id)


def tinylidar_il_single_100():
    """Single-scale model trained exclusively at 1.00x."""
    test_id = "benchmark_tiny_il_single1.00"
    print(test_id)
    planner = TinyLidarNet(
        test_id, 1, 0,
        _tln_path('single1.00'),
        scale=1.0,
        resolution_scales=SINGLE_100_SCALES,
    )
    test_mapless_all_maps(planner, test_id, number_of_laps=NUMBER_OF_LAPS)
    if PLOT:
        plot_trajectory_analysis(planner.name, test_id)


def tinylidar_il_single_075():
    """Single-scale model trained exclusively at 0.75x."""
    test_id = "benchmark_tiny_il_single0.75"
    print(test_id)
    planner = TinyLidarNet(
        test_id, 1, 0,
        _tln_path('single0.75'),
        scale=0.75,
        resolution_scales=SINGLE_075_SCALES,
    )
    test_mapless_all_maps(planner, test_id, number_of_laps=NUMBER_OF_LAPS)
    if PLOT:
        plot_trajectory_analysis(planner.name, test_id)


# Back-compat shim: old callers that used `tinylidar_il_a(scale)` against
# the original multires-shared export still work.
def tinylidar_il_a(scale):
    tinylidar_il_multires_shared(scale)


if __name__ == "__main__":
    # Multires models: evaluate at both scales they were trained on.
    for s in MULTIRES_SCALES:
        tinylidar_il_multires_shared(s)
        tinylidar_il_multires_perbank(s)

    # Single-scale models: one run each, at their native resolution.
    tinylidar_il_single_100()
    tinylidar_il_single_075()
