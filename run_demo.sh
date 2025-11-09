#!/bin/bash

echo "=========================================="
echo "Town Hall Visualization Demo"
echo "=========================================="
echo ""

cd /home/ubuntu/repos/TownVotingSimulator-NeoHackathon

echo "Step 1: Starting the backend server..."
echo "This will run the FastAPI server on port 8000"
echo ""

python main.py &
BACKEND_PID=$!
echo "Backend server started (PID: $BACKEND_PID)"
echo "Waiting for backend to be ready..."
sleep 5

echo ""
echo "Step 2: Generating example town hall meeting..."
echo "Calling /townhall endpoint to create a trajectory..."
echo ""

curl -X POST "http://localhost:8000/townhall" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "immigration and budget policies",
    "num_rounds": 2
  }' > /tmp/townhall_response.json

echo ""
echo "Town hall simulation complete!"
echo ""

TRAJECTORY_FILE=$(cat /tmp/townhall_response.json | grep -o '"trajectory_file":"[^"]*"' | cut -d'"' -f4)
echo "Trajectory file created: $TRAJECTORY_FILE"
echo ""

echo "Step 3: Starting the visualizer..."
echo "The visualizer will be available at http://localhost:5173"
echo ""

cd visualizer/townhall-visualizer
npm run dev &
VISUALIZER_PID=$!

echo ""
echo "=========================================="
echo "Demo is ready!"
echo "=========================================="
echo ""
echo "1. Open your browser to: http://localhost:5173"
echo "2. Upload the trajectory file: trajectories/$TRAJECTORY_FILE"
echo "3. Click 'Play' to watch the town hall meeting visualization"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

wait
