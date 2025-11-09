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

interface GridPosition {
  gridX: number
  gridY: number
  pixelX: number
  pixelY: number
  path: {x: number, y: number}[]
  targetGridX: number
  targetGridY: number
  direction: 'down' | 'left' | 'right' | 'up'
  animFrame: number
  isMoving: boolean
}

const CELL_SIZE = 32
const GRID_COLS = Math.floor(700 / CELL_SIZE)
const GRID_ROWS = Math.floor(400 / CELL_SIZE)
const SPRITE_FRAME_WIDTH = 32
const SPRITE_FRAME_HEIGHT = 32
const LECTERN = { gridX: Math.floor(21 / 2), gridY: 3 }
const getLecternSpot = () => ({ x: LECTERN.gridX, y: LECTERN.gridY - 1 })

function App() {
  const [trajectory, setTrajectory] = useState<Trajectory | null>(null)
  const [currentSpeechIndex, setCurrentSpeechIndex] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [showFinalState, setShowFinalState] = useState(false)
  const [characterPositions, setCharacterPositions] = useState<Record<string, GridPosition>>({})
  const [speakerState, setSpeakerState] = useState<'moving_to_lectern' | 'speaking' | 'leaving_lectern' | 'done'>('done')
  const [lecternOccupiedBy, setLecternOccupiedBy] = useState<string | null>(null)
  const animationFrameRef = useRef<number>()
  const lastMoveTimeRef = useRef<number>(Date.now())
  const lastAnimFrameTimeRef = useRef<number>(Date.now())
  const speechTimerRef = useRef<number | null>(null)
  const plannedMoveToLecternRef = useRef(false)
  const plannedExitRef = useRef(false)
  const speakingTimerStartedRef = useRef(false)
  const speakerStateRef = useRef(speakerState)
  const lecternOccupiedByRef = useRef(lecternOccupiedBy)

  useEffect(() => {
    speakerStateRef.current = speakerState
  }, [speakerState])

  useEffect(() => {
    lecternOccupiedByRef.current = lecternOccupiedBy
  }, [lecternOccupiedBy])

  const loadTrajectory = async (file: File) => {
    const text = await file.text()
    const data: Trajectory = JSON.parse(text)
    setTrajectory(data)
    setCurrentSpeechIndex(0)
    setShowFinalState(false)

    const positions: Record<string, GridPosition> = {}
    const occupiedCells = new Set<string>()

    data.characters.forEach((char, idx) => {
      let gridX = 1 + (idx % (GRID_COLS - 2))
      let gridY = 1 + Math.floor(idx / (GRID_COLS - 2))

      while (occupiedCells.has(`${gridX},${gridY}`)) {
        gridX++
        if (gridX >= GRID_COLS - 1) {
          gridX = 1
          gridY++
        }
      }

      occupiedCells.add(`${gridX},${gridY}`)

      positions[char.id] = {
        gridX,
        gridY,
        pixelX: 50 + gridX * CELL_SIZE + CELL_SIZE / 2,
        pixelY: 50 + gridY * CELL_SIZE + CELL_SIZE / 2,
        path: [],
        targetGridX: gridX,
        targetGridY: gridY,
        direction: 'down',
        animFrame: 1,
        isMoving: false
      }
    })
    setCharacterPositions(positions)
  }

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      loadTrajectory(file)
    }
  }

  const bfs = (startX: number, startY: number, endX: number, endY: number, occupied: Set<string>): {x: number, y: number}[] => {
    const queue: {x: number, y: number, path: {x: number, y: number}[]}[] = [{x: startX, y: startY, path: []}]
    const visited = new Set<string>([`${startX},${startY}`])

    while (queue.length > 0) {
      const current = queue.shift()!

      if (current.x === endX && current.y === endY) {
        return current.path
      }

      const neighbors = [
        {x: current.x, y: current.y - 1},
        {x: current.x - 1, y: current.y},
        {x: current.x + 1, y: current.y},
        {x: current.x, y: current.y + 1}
      ]

      for (const next of neighbors) {
        const key = `${next.x},${next.y}`
        if (next.x >= 1 && next.x < GRID_COLS - 1 &&
            next.y >= 1 && next.y < GRID_ROWS - 1 &&
            !visited.has(key) &&
            !occupied.has(key)) {
          visited.add(key)
          queue.push({
            x: next.x,
            y: next.y,
            path: [...current.path, next]
          })
        }
      }
    }

    return []
  }

  const updateCharacterPositions = () => {
    if (!trajectory) return

    const now = Date.now()
    const shouldMove = now - lastMoveTimeRef.current > 250
    const shouldAnimFrame = now - lastAnimFrameTimeRef.current > 150

    if (shouldMove) {
      lastMoveTimeRef.current = now

      setCharacterPositions(prev => {
        const newPositions = { ...prev }
        const occupiedCells = new Set<string>()

        Object.values(newPositions).forEach(pos => {
          occupiedCells.add(`${pos.gridX},${pos.gridY}`)
        })

        trajectory.characters.forEach(char => {
          const pos = newPositions[char.id]
          if (!pos) return

          const isCurrentSpeaker = lecternOccupiedByRef.current === char.id
          
          if (isCurrentSpeaker && pos.path.length > 0) {
            console.log('SPEAKER MOVING:', char.id, 'at', pos.gridX, pos.gridY, 'target', pos.targetGridX, pos.targetGridY, 'path length', pos.path.length, 'speakerState', speakerStateRef.current)
          }
          
          if (char.id !== lecternOccupiedByRef.current || speakerStateRef.current !== 'moving_to_lectern') {
            occupiedCells.add(`${LECTERN.gridX},${LECTERN.gridY}`)
          }

          if (pos.path.length === 0) {
            if (!isCurrentSpeaker && speakerStateRef.current !== 'moving_to_lectern' && speakerStateRef.current !== 'speaking' && speakerStateRef.current !== 'leaving_lectern' && Math.random() < 0.3) {
              const targetX = 1 + Math.floor(Math.random() * (GRID_COLS - 2))
              const targetY = 1 + Math.floor(Math.random() * (GRID_ROWS - 2))

              occupiedCells.delete(`${pos.gridX},${pos.gridY}`)
              const path = bfs(pos.gridX, pos.gridY, targetX, targetY, occupiedCells)
              occupiedCells.add(`${pos.gridX},${pos.gridY}`)

              if (path.length > 0) {
                pos.path = path
                pos.targetGridX = targetX
                pos.targetGridY = targetY
                pos.isMoving = true
              }
            }
          }else {
            const nextCell = pos.path[0]
            const nextKey = `${nextCell.x},${nextCell.y}`

            occupiedCells.delete(`${pos.gridX},${pos.gridY}`)

            if (!occupiedCells.has(nextKey)) {
              const dx = nextCell.x - pos.gridX
              const dy = nextCell.y - pos.gridY

              if (dy < 0) pos.direction = 'up'
              else if (dy > 0) pos.direction = 'down'
              else if (dx < 0) pos.direction = 'left'
              else if (dx > 0) pos.direction = 'right'

              pos.gridX = nextCell.x
              pos.gridY = nextCell.y
              pos.pixelX = 50 + pos.gridX * CELL_SIZE + CELL_SIZE / 2
              pos.pixelY = 50 + pos.gridY * CELL_SIZE + CELL_SIZE / 2
              pos.path.shift()

              if (pos.path.length === 0) {
                pos.isMoving = false
                pos.animFrame = 1
              }
            } else {
              occupiedCells.delete(`${nextCell.x},${nextCell.y}`)
              const newPath = bfs(pos.gridX, pos.gridY, pos.targetGridX, pos.targetGridY, occupiedCells)
              occupiedCells.add(`${nextCell.x},${nextCell.y}`)
              pos.path = newPath
              if (newPath.length === 0) {
                pos.isMoving = false
                pos.animFrame = 1
              }
            }

            occupiedCells.add(`${pos.gridX},${pos.gridY}`)
          }

          const spot = getLecternSpot()
          if (isCurrentSpeaker) {
            console.log('CHECKING ARRIVAL:', char.id, 'pos:', pos.gridX, pos.gridY, 'spot:', spot.x, spot.y, 'speakerState:', speakerStateRef.current, 'match:', pos.gridX === spot.x && pos.gridY === spot.y)
          }
          if (speakerStateRef.current === 'moving_to_lectern' && isCurrentSpeaker && 
              pos.gridX === spot.x && pos.gridY === spot.y) {
            console.log('✅ ARRIVED AT LECTERN:', char.id, 'at', pos.gridX, pos.gridY)
            pos.path = []
            pos.isMoving = false
            pos.animFrame = 1
            pos.direction = 'down'
            pos.targetGridX = pos.gridX
            pos.targetGridY = pos.gridY
            setSpeakerState('speaking')
          }
          
          if (speakerStateRef.current === 'leaving_lectern' && isCurrentSpeaker && 
              (pos.gridX !== spot.x || pos.gridY !== spot.y)) {
            setLecternOccupiedBy(null)
            setSpeakerState('done')
            setCurrentSpeechIndex(prev => prev + 1)
          }
        })

        return newPositions
      })
    }

    if (shouldAnimFrame) {
      lastAnimFrameTimeRef.current = now

      setCharacterPositions(prev => {
        const newPositions = { ...prev }

        trajectory.characters.forEach(char => {
          const pos = newPositions[char.id]
          if (!pos) return

          if (lecternOccupiedByRef.current === char.id && speakerStateRef.current === 'speaking') {
            pos.animFrame = 1
            pos.isMoving = false
          } else if (pos.isMoving) {
            pos.animFrame = (pos.animFrame + 1) % 3
          } else {
            pos.animFrame = 1
          }
        })

        return newPositions
      })
    }
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
      setSpeakerState('done')
      setLecternOccupiedBy(null)
      return
    }

    const speakerId = trajectory.speeches[currentSpeechIndex].character_id
    setLecternOccupiedBy(speakerId)
    setSpeakerState('moving_to_lectern')
    console.log("Character " + speakerId + " is moving to the lectern")
    plannedMoveToLecternRef.current = false
    plannedExitRef.current = false
    speakingTimerStartedRef.current = false
  }, [isPlaying, currentSpeechIndex, trajectory])

  useEffect(() => {
    if (!isPlaying || !trajectory) return
    const speech = trajectory.speeches[currentSpeechIndex]
    if (!speech) return
    const speakerId = speech.character_id

    if (speakerState === 'moving_to_lectern' && !plannedMoveToLecternRef.current) {
      setCharacterPositions(prev => {
        const newPositions = { ...prev }
        const pos = newPositions[speakerId]
        if (!pos) return prev

        const occupiedCells = new Set<string>()
        Object.values(newPositions).forEach(p => {
          if (p !== pos) occupiedCells.add(`${p.gridX},${p.gridY}`)
        })

        const spot = getLecternSpot()
        console.log('PLANNING PATH TO LECTERN:', speakerId, 'from', pos.gridX, pos.gridY, 'to', spot.x, spot.y)
        const path = bfs(pos.gridX, pos.gridY, spot.x, spot.y, occupiedCells)
        console.log('PATH LENGTH:', path.length, 'PATH:', path)
        if (path.length > 0) {
          pos.path = path
          pos.targetGridX = spot.x
          pos.targetGridY = spot.y
          pos.isMoving = true
        }

        return newPositions
      })
      plannedMoveToLecternRef.current = true
    }

    if (speakerState === 'speaking' && !speakingTimerStartedRef.current) {
      const words = speech.message.trim().split(/\s+/).filter(Boolean).length
      const durationMs = Math.max(2500, Math.min(15000, words * 300))

      if (speechTimerRef.current) clearTimeout(speechTimerRef.current)
      speechTimerRef.current = window.setTimeout(() => {
        setSpeakerState('leaving_lectern')
      }, durationMs)
      speakingTimerStartedRef.current = true
    }

    if (speakerState === 'leaving_lectern' && !plannedExitRef.current) {
      setCharacterPositions(prev => {
        const newPositions = { ...prev }
        const pos = newPositions[speakerId]
        if (!pos) return prev

        if (pos.path.length === 0) {
          const occupiedCells = new Set<string>()
          Object.values(newPositions).forEach(p => {
            if (p !== pos) occupiedCells.add(`${p.gridX},${p.gridY}`)
          })
          occupiedCells.add(`${LECTERN.gridX},${LECTERN.gridY}`)

          let targetX, targetY
          do {
            targetX = 1 + Math.floor(Math.random() * (GRID_COLS - 2))
            targetY = 1 + Math.floor(Math.random() * (GRID_ROWS - 2))
          } while (Math.abs(targetX - LECTERN.gridX) < 2 && Math.abs(targetY - LECTERN.gridY) < 2)

          const path = bfs(pos.gridX, pos.gridY, targetX, targetY, occupiedCells)
          if (path.length > 0) {
            pos.path = path
            pos.targetGridX = targetX
            pos.targetGridY = targetY
            pos.isMoving = true
          }
        }

        return newPositions
      })
      plannedExitRef.current = true
    }
  }, [speakerState, isPlaying, trajectory, currentSpeechIndex])

  const handlePlay = () => {
    setIsPlaying(true)
    if (currentSpeechIndex >= (trajectory?.speeches.length || 0)) {
      setCurrentSpeechIndex(0)
      setShowFinalState(false)
    }
  }

  const handlePause = () => {
    setIsPlaying(false)
    if (speechTimerRef.current) {
      clearTimeout(speechTimerRef.current)
      speechTimerRef.current = null
    }
  }

  const handleReset = () => {
    setCurrentSpeechIndex(0)
    setIsPlaying(false)
    setShowFinalState(false)
    setSpeakerState('done')
    setLecternOccupiedBy(null)
    if (speechTimerRef.current) {
      clearTimeout(speechTimerRef.current)
      speechTimerRef.current = null
    }
  }

  const currentSpeech = trajectory?.speeches[currentSpeechIndex]
  const currentCharacter = trajectory?.characters.find(c => c.id === currentSpeech?.character_id)

  const getDirectionRow = (direction: string): number => {
    switch (direction) {
      case 'down': return 0
      case 'left': return 1
      case 'right': return 2
      case 'up': return 3
      default: return 0
    }
  }

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

          <div className="flex-1 relative" style={{ backgroundColor: '#3a2a1a' }}>
            <div
              className="absolute"
              style={{
                left: '50px',
                top: '50px',
                width: `${GRID_COLS * CELL_SIZE}px`,
                height: `${GRID_ROWS * CELL_SIZE}px`,
                backgroundImage: 'url(/tiles/floor_wood.png)',
                backgroundSize: '32px 32px',
                backgroundPosition: '0 0',
                imageRendering: 'pixelated'
              }}
            />
            <svg className="absolute inset-0 w-full h-full pointer-events-none">
              <text x="400" y="30" textAnchor="middle" fill="white" fontSize="20" fontWeight="bold">
                Town Hall
              </text>
            </svg>

            <div
              className="absolute"
              style={{
                left: `${50 + LECTERN.gridX * CELL_SIZE + CELL_SIZE / 2}px`,
                top: `${50 + LECTERN.gridY * CELL_SIZE + CELL_SIZE / 2}px`,
                width: `${CELL_SIZE}px`,
                height: `${CELL_SIZE}px`,
                transform: 'translate(-50%, -50%)',
                backgroundImage: 'url(/lectern.png)',
                backgroundSize: '100% 100%',
                imageRendering: 'pixelated',
                zIndex: 1
              }}
            />

            {trajectory.characters.map(char => {
              const pos = characterPositions[char.id]
              if (!pos) return null

              const isSpeaking = currentSpeech?.character_id === char.id
              const directionRow = getDirectionRow(pos.direction)
              const frameCol = pos.animFrame

              const debugBorder =
                char.id === lecternOccupiedByRef.current && speakerStateRef.current === 'speaking' ? '3px solid #22c55e' :
                (char.id === lecternOccupiedByRef.current && speakerStateRef.current === 'moving_to_lectern') || pos.isMoving ? '3px solid #ef4444' :
                'none'

              return (
                <div
                  key={char.id}
                  className="absolute"
                  style={{
                    left: `${pos.pixelX}px`,
                    top: `${pos.pixelY}px`,
                    width: `${SPRITE_FRAME_WIDTH}px`,
                    height: `${SPRITE_FRAME_HEIGHT}px`,
                    transform: 'translate(-50%, -50%)'
                  }}
                >
                  <div className="relative">
                    <div
                      style={{
                        width: `${SPRITE_FRAME_WIDTH}px`,
                        height: `${SPRITE_FRAME_HEIGHT}px`,
                        backgroundImage: `url(/characters/${char.sprite})`,
                        backgroundPosition: `-${frameCol * SPRITE_FRAME_WIDTH}px -${directionRow * SPRITE_FRAME_HEIGHT}px`,
                        backgroundRepeat: 'no-repeat',
                        imageRendering: 'pixelated',
                        border: debugBorder,
                        borderRadius: '4px'
                      }}
                    />
                    {isSpeaking && currentSpeech && (
                      <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 bg-white/90 text-black px-2 py-1 rounded-full text-base whitespace-nowrap shadow-lg">
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
                <div
                  style={{
                    width: `${SPRITE_FRAME_WIDTH * 2}px`,
                    height: `${SPRITE_FRAME_HEIGHT * 2}px`,
                    backgroundImage: `url(/characters/${currentCharacter.sprite})`,
                    backgroundPosition: `-${SPRITE_FRAME_WIDTH * 1 * 2}px -${SPRITE_FRAME_HEIGHT * 0 * 2}px`,
                    backgroundSize: `${SPRITE_FRAME_WIDTH * 3 * 2}px ${SPRITE_FRAME_HEIGHT * 4 * 2}px`,
                    backgroundRepeat: 'no-repeat',
                    imageRendering: 'pixelated'
                  }}
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
