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

const POLITICIAN_LECTERNS = [
  { gridX: 5, gridY: 2, speakerSpot: { x: 5, y: 1 } },
  { gridX: 15, gridY: 2, speakerSpot: { x: 15, y: 1 } }
]
const AUDIENCE_LECTERN = { gridX: 10, gridY: 5, speakerSpot: { x: 10, y: 6 } }
const STAGE_ROWS = [1, 2, 3]
const STAGE_MARGIN_COLS = 2
const AUDIENCE_SECTION = { startY: 7, endY: 11 }
const AUDIENCE_MIN_Y = Math.max(AUDIENCE_SECTION.startY, AUDIENCE_LECTERN.speakerSpot.y + 1)

const CENTER_ZONE = { startX: AUDIENCE_LECTERN.gridX - 2, endX: AUDIENCE_LECTERN.gridX + 2 }
const LEFT_ZONE = { startX: STAGE_MARGIN_COLS, endX: CENTER_ZONE.startX - 1 }
const RIGHT_ZONE = { startX: CENTER_ZONE.endX + 1, endX: GRID_COLS - 1 - STAGE_MARGIN_COLS }

const getPoliticianLectern = (charId: string) => {
  if (charId === 'politician_1') return POLITICIAN_LECTERNS[0]
  if (charId === 'politician_2') return POLITICIAN_LECTERNS[1]
  return null
}

const getSpeakingSpot = (char: Character) => {
  const politicianLectern = getPoliticianLectern(char.id)
  if (politicianLectern) return politicianLectern.speakerSpot
  return AUDIENCE_LECTERN.speakerSpot
}

const findAvailablePositionNear = (
  targetX: number, 
  targetY: number, 
  radius: number, 
  occupiedCells: Set<string>, 
  currentX: number, 
  currentY: number,
  bounds?: { minX: number, maxX: number, minY: number, maxY: number }
): {x: number, y: number} | null => {
  const candidates: {x: number, y: number, distance: number}[] = []
  
  const minX = bounds?.minX ?? 1
  const maxX = bounds?.maxX ?? GRID_COLS - 2
  const minY = bounds?.minY ?? 1
  const maxY = bounds?.maxY ?? GRID_ROWS - 2
  
  for (let dx = -radius; dx <= radius; dx++) {
    for (let dy = -radius; dy <= radius; dy++) {
      const x = targetX + dx
      const y = targetY + dy
      const key = `${x},${y}`
      
      if (x >= minX && x <= maxX && y >= minY && y <= maxY && !occupiedCells.has(key)) {
        const distance = Math.abs(x - currentX) + Math.abs(y - currentY)
        candidates.push({x, y, distance})
      }
    }
  }
  
  if (candidates.length === 0) return null
  
  candidates.sort((a, b) => a.distance - b.distance)
  return {x: candidates[0].x, y: candidates[0].y}
}

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
  
  const [typingText, setTypingText] = useState('')
  const typingIndexRef = useRef(0)
  const typingTimerRef = useRef<number | null>(null)
  const holdTimerRef = useRef<number | null>(null)
  const typingStartedRef = useRef(false)
  
  const TYPING_CPS = 30
  const HOLD_MS = 2000

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
      let gridX, gridY, direction
      
      const politicianLectern = getPoliticianLectern(char.id)
      if (politicianLectern) {
        gridX = politicianLectern.speakerSpot.x
        gridY = politicianLectern.speakerSpot.y
        direction = 'down'
      } else {
        const audienceIdx = idx - 2
        const cols = GRID_COLS - 4
        gridX = 2 + (audienceIdx % cols)
        gridY = AUDIENCE_SECTION.startY + Math.floor(audienceIdx / cols)
        direction = 'up'
        
        while (occupiedCells.has(`${gridX},${gridY}`) || gridY > AUDIENCE_SECTION.endY) {
          gridX++
          if (gridX >= GRID_COLS - 2) {
            gridX = 2
            gridY++
          }
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
        direction: direction as 'down' | 'left' | 'right' | 'up',
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

        const sortedCharacters = [...trajectory.characters].sort((a, b) => {
          const aIsSpeaker = a.id === lecternOccupiedByRef.current
          const bIsSpeaker = b.id === lecternOccupiedByRef.current
          if (aIsSpeaker && !bIsSpeaker) return -1
          if (!aIsSpeaker && bIsSpeaker) return 1
          return 0
        })

        sortedCharacters.forEach(char => {
          const pos = newPositions[char.id]
          if (!pos) return

          const isCurrentSpeaker = lecternOccupiedByRef.current === char.id
          
          if (isCurrentSpeaker && pos.path.length > 0) {
            console.log('SPEAKER MOVING:', char.id, 'at', pos.gridX, pos.gridY, 'target', pos.targetGridX, pos.targetGridY, 'path length', pos.path.length, 'speakerState', speakerStateRef.current)
          }
          
          POLITICIAN_LECTERNS.forEach(lectern => {
            occupiedCells.add(`${lectern.gridX},${lectern.gridY}`)
          })
          occupiedCells.add(`${AUDIENCE_LECTERN.gridX},${AUDIENCE_LECTERN.gridY}`)
          
          if (char.id !== lecternOccupiedByRef.current && (speakerStateRef.current === 'moving_to_lectern' || speakerStateRef.current === 'speaking')) {
            const currentSpeaker = trajectory.characters.find(c => c.id === lecternOccupiedByRef.current)
            if (currentSpeaker) {
              const spot = getSpeakingSpot(currentSpeaker)
              occupiedCells.add(`${spot.x},${spot.y}`)
            }
          }

          if (pos.path.length === 0) {
            const isPolitician = getPoliticianLectern(char.id) !== null
            
            if (!isPolitician && !isCurrentSpeaker) {
              let swarmTarget: {x: number, y: number} | null = null
              let swarmZone: { startX: number, endX: number } | null = null
              
              if (speakerStateRef.current === 'speaking' || speakerStateRef.current === 'moving_to_lectern') {
                const currentSpeaker = trajectory.characters.find(c => c.id === lecternOccupiedByRef.current)
                if (currentSpeaker) {
                  const speakerPoliticianLectern = getPoliticianLectern(currentSpeaker.id)
                  if (speakerPoliticianLectern) {
                    if (currentSpeaker.id === 'politician_1') {
                      swarmZone = LEFT_ZONE
                      swarmTarget = {
                        x: Math.max(LEFT_ZONE.startX, Math.min(LEFT_ZONE.endX, speakerPoliticianLectern.gridX)),
                        y: AUDIENCE_MIN_Y
                      }
                    } else if (currentSpeaker.id === 'politician_2') {
                      swarmZone = RIGHT_ZONE
                      swarmTarget = {
                        x: Math.max(RIGHT_ZONE.startX, Math.min(RIGHT_ZONE.endX, speakerPoliticianLectern.gridX)),
                        y: AUDIENCE_MIN_Y
                      }
                    }
                  } else {
                    swarmZone = CENTER_ZONE
                    swarmTarget = {x: AUDIENCE_LECTERN.gridX, y: AUDIENCE_MIN_Y}
                  }
                }
              }
              
              if (swarmTarget && swarmZone && Math.random() < 0.4) {
                occupiedCells.delete(`${pos.gridX},${pos.gridY}`)
                
                const localOccupied = new Set(occupiedCells)
                for (let x = 1; x < GRID_COLS - 1; x++) {
                  for (let y = 1; y < AUDIENCE_MIN_Y; y++) {
                    localOccupied.add(`${x},${y}`)
                  }
                }
                
                const availablePos = findAvailablePositionNear(
                  swarmTarget.x, 
                  swarmTarget.y, 
                  4, 
                  localOccupied, 
                  pos.gridX, 
                  pos.gridY,
                  {
                    minX: swarmZone.startX,
                    maxX: swarmZone.endX,
                    minY: AUDIENCE_MIN_Y,
                    maxY: AUDIENCE_SECTION.endY
                  }
                )
                
                if (availablePos) {
                  occupiedCells.add(`${availablePos.x},${availablePos.y}`)
                  
                  const path = bfs(pos.gridX, pos.gridY, availablePos.x, availablePos.y, localOccupied)
                  if (path.length > 0) {
                    pos.path = path
                    pos.targetGridX = availablePos.x
                    pos.targetGridY = availablePos.y
                    pos.isMoving = true
                  } else {
                    occupiedCells.delete(`${availablePos.x},${availablePos.y}`)
                  }
                }
                
                occupiedCells.add(`${pos.gridX},${pos.gridY}`)
              } else if (speakerStateRef.current !== 'moving_to_lectern' && speakerStateRef.current !== 'speaking' && speakerStateRef.current !== 'leaving_lectern' && Math.random() < 0.2) {
                occupiedCells.delete(`${pos.gridX},${pos.gridY}`)
                
                const localOccupied = new Set(occupiedCells)
                for (let x = 1; x < GRID_COLS - 1; x++) {
                  for (let y = 1; y < AUDIENCE_MIN_Y; y++) {
                    localOccupied.add(`${x},${y}`)
                  }
                }
                
                const cols = GRID_COLS - 4
                const targetX = 2 + Math.floor(Math.random() * cols)
                const targetY = AUDIENCE_MIN_Y + Math.floor(Math.random() * (AUDIENCE_SECTION.endY - AUDIENCE_MIN_Y + 1))

                const path = bfs(pos.gridX, pos.gridY, targetX, targetY, localOccupied)
                occupiedCells.add(`${pos.gridX},${pos.gridY}`)

                if (path.length > 0) {
                  pos.path = path
                  pos.targetGridX = targetX
                  pos.targetGridY = targetY
                  pos.isMoving = true
                }
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
              const newPath = bfs(pos.gridX, pos.gridY, pos.targetGridX, pos.targetGridY, occupiedCells)
              if (newPath.length > 0) {
                pos.path = newPath
              } else {
                pos.path = []
                pos.isMoving = false
                pos.animFrame = 1
              }
            }

            occupiedCells.add(`${pos.gridX},${pos.gridY}`)
          }

          const spot = getSpeakingSpot(char)
          if (isCurrentSpeaker) {
            console.log('CHECKING ARRIVAL:', char.id, 'pos:', pos.gridX, pos.gridY, 'spot:', spot.x, spot.y, 'speakerState:', speakerStateRef.current, 'match:', pos.gridX === spot.x && pos.gridY === spot.y)
          }
          if (speakerStateRef.current === 'moving_to_lectern' && isCurrentSpeaker && 
              pos.gridX === spot.x && pos.gridY === spot.y) {
            console.log('✅ ARRIVED AT LECTERN:', char.id, 'at', pos.gridX, pos.gridY)
            pos.path = []
            pos.isMoving = false
            pos.animFrame = 1
            const isPolitician = getPoliticianLectern(char.id) !== null
            pos.direction = isPolitician ? 'down' : 'up'
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
            const isPolitician = getPoliticianLectern(char.id) !== null
            if (!isPolitician && !pos.isMoving && char.id !== lecternOccupiedByRef.current) {
              pos.direction = 'up'
            }
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

        const char = trajectory.characters.find(c => c.id === speakerId)
        if (!char) return prev
        const spot = getSpeakingSpot(char)
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
          POLITICIAN_LECTERNS.forEach(lectern => {
            occupiedCells.add(`${lectern.gridX},${lectern.gridY}`)
          })
          occupiedCells.add(`${AUDIENCE_LECTERN.gridX},${AUDIENCE_LECTERN.gridY}`)

          const cols = GRID_COLS - 4
          const targetX = 2 + Math.floor(Math.random() * cols)
          const targetY = AUDIENCE_SECTION.startY + Math.floor(Math.random() * (AUDIENCE_SECTION.endY - AUDIENCE_SECTION.startY + 1))

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

  useEffect(() => {
    if (!isPlaying || !trajectory || speakerState !== 'speaking') return
    const speech = trajectory.speeches[currentSpeechIndex]
    if (!speech) return
    const speakerId = speech.character_id
    const char = trajectory.characters.find(c => c.id === speakerId)
    if (!char) return

    if (!typingStartedRef.current) {
      typingStartedRef.current = true
      setTypingText('')
      typingIndexRef.current = 0

      const message = speech.message
      const isPolitician = getPoliticianLectern(char.id) !== null

      const typeNextChar = () => {
        if (typingIndexRef.current < message.length) {
          const nextChar = message[typingIndexRef.current]
          setTypingText(prev => prev + nextChar)
          typingIndexRef.current++

          const isPunctuation = /[.!?,;:]/.test(nextChar)
          const delay = isPunctuation ? 1000 / (TYPING_CPS / 3) : 1000 / TYPING_CPS

          typingTimerRef.current = window.setTimeout(typeNextChar, delay)
        } else {
          holdTimerRef.current = window.setTimeout(() => {
            if (isPolitician) {
              setSpeakerState('done')
              setCurrentSpeechIndex(prev => prev + 1)
            } else {
              setSpeakerState('leaving_lectern')
            }
          }, HOLD_MS)
        }
      }

      typeNextChar()
    }

    return () => {
      if (typingTimerRef.current) {
        clearTimeout(typingTimerRef.current)
        typingTimerRef.current = null
      }
      if (holdTimerRef.current) {
        clearTimeout(holdTimerRef.current)
        holdTimerRef.current = null
      }
    }
  }, [speakerState, isPlaying, trajectory, currentSpeechIndex, TYPING_CPS, HOLD_MS])

  useEffect(() => {
    typingStartedRef.current = false
    setTypingText('')
    typingIndexRef.current = 0
  }, [currentSpeechIndex])

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
    if (typingTimerRef.current) {
      clearTimeout(typingTimerRef.current)
      typingTimerRef.current = null
    }
    if (holdTimerRef.current) {
      clearTimeout(holdTimerRef.current)
      holdTimerRef.current = null
    }
  }

  const handleReset = () => {
    setCurrentSpeechIndex(0)
    setIsPlaying(false)
    setShowFinalState(false)
    setSpeakerState('done')
    setLecternOccupiedBy(null)
    setTypingText('')
    typingIndexRef.current = 0
    typingStartedRef.current = false
    if (speechTimerRef.current) {
      clearTimeout(speechTimerRef.current)
      speechTimerRef.current = null
    }
    if (typingTimerRef.current) {
      clearTimeout(typingTimerRef.current)
      typingTimerRef.current = null
    }
    if (holdTimerRef.current) {
      clearTimeout(holdTimerRef.current)
      holdTimerRef.current = null
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
            
            <div
              className="absolute"
              style={{
                left: `${50 + STAGE_MARGIN_COLS * CELL_SIZE}px`,
                top: `${50 + STAGE_ROWS[0] * CELL_SIZE}px`,
                width: `${(GRID_COLS - 2 * STAGE_MARGIN_COLS) * CELL_SIZE}px`,
                height: `${STAGE_ROWS.length * CELL_SIZE}px`,
                backgroundColor: '#8B4513',
                border: '2px solid #654321',
                boxShadow: '0 4px 6px rgba(0,0,0,0.3)',
                zIndex: 0
              }}
            />
            
            <svg className="absolute inset-0 w-full h-full pointer-events-none">
              <text x="400" y="30" textAnchor="middle" fill="white" fontSize="20" fontWeight="bold">
                Town Hall
              </text>
            </svg>

            {POLITICIAN_LECTERNS.map((lectern, idx) => (
              <div
                key={`politician-lectern-${idx}`}
                className="absolute"
                style={{
                  left: `${50 + lectern.gridX * CELL_SIZE + CELL_SIZE / 2}px`,
                  top: `${50 + lectern.gridY * CELL_SIZE + CELL_SIZE / 2}px`,
                  width: `${CELL_SIZE * 1.25}px`,
                  height: `${CELL_SIZE * 1.25}px`,
                  transform: 'translate(-50%, -50%)',
                  backgroundImage: 'url(/lectern.png)',
                  backgroundSize: '100% 100%',
                  imageRendering: 'pixelated',
                  zIndex: 1
                }}
              />
            ))}
            
            <div
              className="absolute"
              style={{
                left: `${50 + AUDIENCE_LECTERN.gridX * CELL_SIZE + CELL_SIZE / 2}px`,
                top: `${50 + AUDIENCE_LECTERN.gridY * CELL_SIZE + CELL_SIZE / 2}px`,
                width: `${CELL_SIZE * 1.25}px`,
                height: `${CELL_SIZE * 1.25}px`,
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
                    transform: 'translate(-50%, -50%)',
                    zIndex: 10
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
                      <div className="absolute top-0 left-full ml-2 bg-white/90 text-black px-2 py-1 rounded-full text-base whitespace-nowrap shadow-lg">
                        {currentSpeech.emoji_summary}
                      </div>
                    )}
                  </div>
                  <div className="text-xs text-center mt-1 bg-black bg-opacity-30 px-1 rounded whitespace-nowrap" style={{ position: 'relative', zIndex: 20 }}>
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
                  <p className="text-gray-300">
                    {typingText}
                    {typingIndexRef.current < currentSpeech.message.length && (
                      <span className="animate-pulse">▌</span>
                    )}
                  </p>
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
