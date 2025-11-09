import { useState, useEffect, useRef } from 'react'
import './App.css'

interface Character {
  id: string
  name: string
  role: string
  sprite: string
  initial_position: { x: number; y: number }
}

interface Speech {
  round: number
  character_id: string
  character_name: string
  message: string
  emoji_summary: string
  timestamp: string
  sequence: number
}

interface CharacterInterest {
  character_id: string
  character_name: string
  vote_intent: string
  summary: string
}

interface Trajectory {
  metadata: {
    topic: string
    num_rounds: number
    timestamp: string
    duration_seconds: number
  }
  characters: Character[]
  speeches: Speech[]
  final_state: {
    character_interests: CharacterInterest[]
  }
}

interface CharacterPosition {
  x: number
  y: number
  targetX: number
  targetY: number
}

function App() {
  const [trajectory, setTrajectory] = useState<Trajectory | null>(null)
  const [currentSpeechIndex, setCurrentSpeechIndex] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [showFinalState, setShowFinalState] = useState(false)
  const [characterPositions, setCharacterPositions] = useState<Record<string, CharacterPosition>>({})
  const [fileInput, setFileInput] = useState<File | null>(null)
  const animationFrameRef = useRef<number>()

  const loadTrajectory = async (file: File) => {
    const text = await file.text()
    const data: Trajectory = JSON.parse(text)
    setTrajectory(data)
    setCurrentSpeechIndex(0)
    setShowFinalState(false)
    
    const positions: Record<string, CharacterPosition> = {}
    data.characters.forEach(char => {
      positions[char.id] = {
        x: char.initial_position.x,
        y: char.initial_position.y,
        targetX: char.initial_position.x,
        targetY: char.initial_position.y
      }
    })
    setCharacterPositions(positions)
  }

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setFileInput(file)
      loadTrajectory(file)
    }
  }

  const updateCharacterPositions = () => {
    if (!trajectory) return

    setCharacterPositions(prev => {
      const newPositions = { ...prev }
      
      trajectory.characters.forEach(char => {
        const pos = newPositions[char.id]
        if (!pos) return

        if (Math.random() < 0.02) {
          const townHallBounds = { minX: 50, maxX: 750, minY: 50, maxY: 450 }
          pos.targetX = townHallBounds.minX + Math.random() * (townHallBounds.maxX - townHallBounds.minX)
          pos.targetY = townHallBounds.minY + Math.random() * (townHallBounds.maxY - townHallBounds.minY)
        }

        const dx = pos.targetX - pos.x
        const dy = pos.targetY - pos.y
        const distance = Math.sqrt(dx * dx + dy * dy)

        if (distance > 1) {
          const speed = 0.5
          pos.x += (dx / distance) * speed
          pos.y += (dy / distance) * speed
        }
      })

      return newPositions
    })
  }

  useEffect(() => {
    const animate = () => {
      updateCharacterPositions()
      animationFrameRef.current = requestAnimationFrame(animate)
    }
    
    if (trajectory) {
      animationFrameRef.current = requestAnimationFrame(animate)
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [trajectory])

  useEffect(() => {
    if (!isPlaying || !trajectory) return

    if (currentSpeechIndex >= trajectory.speeches.length) {
      setIsPlaying(false)
      setShowFinalState(true)
      return
    }

    const timer = setTimeout(() => {
      setCurrentSpeechIndex(prev => prev + 1)
    }, 3000)

    return () => clearTimeout(timer)
  }, [isPlaying, currentSpeechIndex, trajectory])

  const handlePlay = () => {
    setIsPlaying(true)
    if (currentSpeechIndex >= (trajectory?.speeches.length || 0)) {
      setCurrentSpeechIndex(0)
      setShowFinalState(false)
    }
  }

  const handlePause = () => {
    setIsPlaying(false)
  }

  const handleReset = () => {
    setCurrentSpeechIndex(0)
    setIsPlaying(false)
    setShowFinalState(false)
  }

  const currentSpeech = trajectory?.speeches[currentSpeechIndex]
  const currentCharacter = trajectory?.characters.find(c => c.id === currentSpeech?.character_id)

  if (!trajectory) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-900">
        <div className="bg-gray-800 p-8 rounded-lg shadow-xl max-w-md w-full">
          <h1 className="text-3xl font-bold text-white mb-6 text-center">Town Hall Visualizer</h1>
          <div className="space-y-4">
            <p className="text-gray-300 text-center">Load a trajectory JSON file to visualize the town hall meeting</p>
            <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-gray-600 border-dashed rounded-lg cursor-pointer bg-gray-700 hover:bg-gray-600 transition-colors">
              <div className="flex flex-col items-center justify-center pt-5 pb-6">
                <svg className="w-10 h-10 mb-3 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <p className="mb-2 text-sm text-gray-400">
                  <span className="font-semibold">Click to upload</span> or drag and drop
                </p>
                <p className="text-xs text-gray-500">JSON trajectory file</p>
              </div>
              <input type="file" className="hidden" accept=".json" onChange={handleFileUpload} />
            </label>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="flex h-screen">
        <div className="flex-1 flex flex-col">
          <div className="bg-gray-800 p-4 border-b border-gray-700">
            <h1 className="text-2xl font-bold">{trajectory.metadata.topic}</h1>
            <div className="flex gap-4 mt-2">
              <button
                onClick={handlePlay}
                disabled={isPlaying}
                className="px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded transition-colors"
              >
                Play
              </button>
              <button
                onClick={handlePause}
                disabled={!isPlaying}
                className="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 disabled:bg-gray-600 rounded transition-colors"
              >
                Pause
              </button>
              <button
                onClick={handleReset}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded transition-colors"
              >
                Reset
              </button>
              <span className="px-4 py-2 bg-gray-700 rounded">
                Speech {currentSpeechIndex + 1} / {trajectory.speeches.length}
              </span>
            </div>
          </div>

          <div className="flex-1 relative bg-gradient-to-br from-green-900 to-green-800">
            <svg className="absolute inset-0 w-full h-full">
              <rect x="50" y="50" width="700" height="400" fill="#8B4513" stroke="#654321" strokeWidth="4" />
              <text x="400" y="30" textAnchor="middle" fill="white" fontSize="20" fontWeight="bold">
                Town Hall
              </text>
            </svg>

            {trajectory.characters.map(char => {
              const pos = characterPositions[char.id]
              if (!pos) return null

              const isSpeaking = currentSpeech?.character_id === char.id

              return (
                <div
                  key={char.id}
                  className="absolute transition-all duration-100"
                  style={{
                    left: `${pos.x}px`,
                    top: `${pos.y}px`,
                    transform: 'translate(-50%, -50%)'
                  }}
                >
                  <div className="relative">
                    <img
                      src={`/characters/${char.sprite}`}
                      alt={char.name}
                      className={`w-12 h-12 ${isSpeaking ? 'ring-4 ring-yellow-400' : ''}`}
                      style={{ imageRendering: 'pixelated' }}
                    />
                    {isSpeaking && currentSpeech && (
                      <div className="absolute -top-12 left-1/2 transform -translate-x-1/2 bg-white text-black px-3 py-1 rounded-full text-2xl whitespace-nowrap shadow-lg">
                        {currentSpeech.emoji_summary}
                      </div>
                    )}
                  </div>
                  <div className="text-xs text-center mt-1 bg-black bg-opacity-50 px-1 rounded whitespace-nowrap">
                    {char.name}
                  </div>
                </div>
              )
            })}
          </div>

          {currentSpeech && currentCharacter && !showFinalState && (
            <div className="bg-gray-800 p-6 border-t border-gray-700">
              <div className="flex items-start gap-4">
                <img
                  src={`/characters/${currentCharacter.sprite}`}
                  alt={currentCharacter.name}
                  className="w-16 h-16"
                  style={{ imageRendering: 'pixelated' }}
                />
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <h3 className="text-xl font-bold">{currentSpeech.character_name}</h3>
                    <span className="text-2xl">{currentSpeech.emoji_summary}</span>
                  </div>
                  <p className="text-gray-300">{currentSpeech.message}</p>
                  <p className="text-sm text-gray-500 mt-2">Round {currentSpeech.round}</p>
                </div>
              </div>
            </div>
          )}

          {showFinalState && (
            <div className="bg-gray-800 p-6 border-t border-gray-700">
              <h2 className="text-2xl font-bold mb-4">Final Voting Interests</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {trajectory.final_state.character_interests.map(interest => (
                  <div key={interest.character_id} className="bg-gray-700 p-4 rounded">
                    <h3 className="font-bold text-lg">{interest.character_name}</h3>
                    <p className="text-sm text-gray-400 mb-2">
                      Vote Intent: <span className="text-yellow-400">{interest.vote_intent || 'Undecided'}</span>
                    </p>
                    <p className="text-sm text-gray-300">{interest.summary}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="w-80 bg-gray-800 border-l border-gray-700 overflow-y-auto">
          <div className="p-4 border-b border-gray-700">
            <h2 className="text-xl font-bold">Message History</h2>
          </div>
          <div className="p-4 space-y-3">
            {trajectory.speeches.map((speech, index) => (
              <div
                key={speech.sequence}
                className={`p-3 rounded ${
                  index === currentSpeechIndex ? 'bg-blue-900' : 'bg-gray-700'
                } ${index <= currentSpeechIndex ? 'opacity-100' : 'opacity-50'}`}
              >
                <div className="flex items-center gap-2 mb-1">
                  <span className="font-bold text-sm">{speech.character_name}</span>
                  <span className="text-lg">{speech.emoji_summary}</span>
                </div>
                <p className="text-xs text-gray-300">{speech.message}</p>
                <p className="text-xs text-gray-500 mt-1">Round {speech.round}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
