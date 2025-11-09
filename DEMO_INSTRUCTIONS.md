# Town Hall Visualization Demo Instructions

## Quick Start

The demo is already running! Here's how to view it:

### 1. Access the Visualizer

Open your browser and go to:
**https://user:be141ebd742b2547cfce9898fb5513cd@townhall-visualization-app-tunnel-k4hec1xf.devinapps.com**

### 2. Load the Trajectory File

The visualizer will show a file upload interface. You need to load the trajectory JSON file that was generated.

**Trajectory file location:** `/home/ubuntu/repos/TownVotingSimulator-NeoHackathon/trajectories/trajectory_20251109_071733.json`

Since you're viewing this in a browser, you'll need to download the trajectory file first:
- The trajectory file is available on the server at the path above
- You can download it and then upload it to the visualizer

### 3. Watch the Visualization

Once the trajectory is loaded:
1. Click the **Play** button to start the animation
2. Watch the characters pace around the town hall rectangle
3. See speech bubbles with emoji summaries appear above speaking characters
4. View the full message in the bottom panel with the character's avatar
5. Check the side menu for the complete message history
6. At the end, see the final voting interests display

### Controls

- **Play**: Start/resume the visualization
- **Pause**: Pause the animation
- **Reset**: Go back to the beginning

## What You're Seeing

- **2D RPG Grid**: A top-down view of the town hall (brown rectangle)
- **Characters**: 7 characters (5 villagers + 2 politicians) with sprites from the Smallville paper
- **Random Pacing**: Characters randomly walk around the town hall (visual effect only)
- **Speech Bubbles**: Emoji summaries appear above the speaking character
- **Bottom Panel**: Shows the current speaker's avatar and full message
- **Side Menu**: Complete chronological history of all messages
- **Final State**: Voting interests and preferences at the end

## Running Your Own Simulation

To generate a new trajectory:

```bash
curl -X POST "http://localhost:8000/townhall" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "your topic here",
    "num_rounds": 2
  }'
```

This will create a new trajectory file in the `trajectories/` directory that you can load in the visualizer.

## Technical Details

- **Backend**: FastAPI server on port 8000
- **Frontend**: React + Vite + Tailwind CSS on port 5173
- **Character Sprites**: From the Stanford Smallville generative agents paper
- **Trajectory Format**: JSON with metadata, characters, speeches, and final state
