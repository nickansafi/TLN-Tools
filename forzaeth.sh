mkdir forzaeth_ws
cd forzaeth_ws
mkdir src
catkin_make
cd src
git clone --recurse-submodules git@github.com:ForzaETH/race_stack.git 
cd race_stack
xargs sudo apt-get install -y < ./.devcontainer/.install_utils/linux_req_sim.txt
pip install -r ./.devcontainer/.install_utils/requirements.txt
pip install ~/catkin_ws/src/race_stack/f110_utils/libs/ccma
pip install -e ~/catkin_ws/src/race_stack/planner/graph_based_planner/src/GraphBasedPlanner
cd ..
catkin build
