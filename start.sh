if [ "$1" = "setup" ]; then
    python3 -m venv venv
    source venv/bin/activate
    pip3 install scikit-learn
    pip3 install tensorflow==2.13
    pip3 install rosbags
    pip3 install pyglet
    cd Benchmark
    pip3 install -r requirements.txt
    pip3 install -e .
    git submodule init
    git submodule update
    cd trajectory_planning_helpers
    pip3 install -e .
elif [ "$1" = "train" ]; then
    source venv/bin/activate
    python3 $2.py
elif [ "$1" = "benchmark" ]; then
    cd "$(dirname "$0")"/
    source venv/bin/activate
    python3 ./Benchmark/f1tenth_benchmarks/benchmark_results/$2.py
fi