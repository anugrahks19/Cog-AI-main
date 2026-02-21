import { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Trophy, RefreshCw, ArrowRight } from "lucide-react";

const PUZZLE_SIZE = 300;
// Using a cute, distinct cat image with clear colors
const IMAGE_URL = "https://images.unsplash.com/photo-1596854407944-bf87f6fdd49e?auto=format&fit=crop&q=80&w=300&h=300";

const MAX_LEVEL = 10;

interface Tile {
    id: number;
    currentPos: number; // 0 to N
    correctPos: number; // 0 to N
}

export default function JigsawPuzzle() {
    const [level, setLevel] = useState(1);
    const [tiles, setTiles] = useState<Tile[]>([]);
    const [selectedTile, setSelectedTile] = useState<number | null>(null);
    const [isComplete, setIsComplete] = useState(false);
    const [moves, setMoves] = useState(0);

    // Level 1 = 3x3, Level 2 = 4x4, etc.
    const gridSize = level + 2;
    const tileSize = PUZZLE_SIZE / gridSize;

    const initializePuzzle = useCallback((currentLevel: number) => {
        const currentGridSize = currentLevel + 2;
        let newTiles: Tile[] = Array.from({ length: currentGridSize * currentGridSize }, (_, i) => ({
            id: i,
            currentPos: i,
            correctPos: i,
        }));

        // Shuffle
        for (let i = newTiles.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            // swap currentPos
            const temp = newTiles[i].currentPos;
            newTiles[i].currentPos = newTiles[j].currentPos;
            newTiles[j].currentPos = temp;
        }

        setTiles(newTiles);
        setSelectedTile(null);
        setIsComplete(false);
        setMoves(0);
    }, []);

    useEffect(() => {
        initializePuzzle(level);
    }, [level, initializePuzzle]);

    const handleTileClick = (index: number) => {
        if (isComplete) return;

        if (selectedTile === null) {
            setSelectedTile(index);
        } else {
            // Swap tiles
            setTiles(prevTiles => {
                const newTiles = [...prevTiles];
                const temp = newTiles[selectedTile].currentPos;
                newTiles[selectedTile].currentPos = newTiles[index].currentPos;
                newTiles[index].currentPos = temp;
                return newTiles;
            });
            setMoves(m => m + 1);
            setSelectedTile(null);
        }
    };

    useEffect(() => {
        if (tiles.length > 0 && tiles.every(t => t.currentPos === t.correctPos)) {
            setIsComplete(true);
        }
    }, [tiles]);

    const handleNextLevel = () => {
        if (level < MAX_LEVEL) {
            setLevel(l => l + 1);
        }
    };

    const handlePlayAgain = () => {
        setLevel(1);
    };

    return (
        <Card className="min-h-[400px] flex flex-col items-center justify-center p-6 bg-slate-50 dark:bg-slate-900 border-none relative overflow-hidden">
            <div className="absolute top-4 left-4 text-xs font-semibold text-primary bg-primary/10 px-2 py-1 rounded">
                Level {level} / {MAX_LEVEL}
            </div>
            <div className="absolute top-4 right-4 text-xs font-semibold text-slate-500 bg-white/50 dark:bg-slate-800/50 px-2 py-1 rounded">
                Moves: {moves}
            </div>

            <div className="text-center space-y-4 mb-6 z-10 w-full flex flex-col items-center mt-6">
                <div className="flex flex-col md:flex-row items-center gap-6 mb-2">
                    <div className="flex flex-col items-center md:items-start text-center md:text-left">
                        <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">Jigsaw Puzzle</h3>
                        <p className="text-sm text-slate-600 dark:text-slate-400 max-w-xs">
                            Swap the pieces to reveal the complete image. Select one piece, then select another to swap them.
                        </p>
                    </div>

                    {!isComplete && (
                        <div className="flex flex-col items-center bg-white dark:bg-slate-800 p-2 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 shrink-0">
                            <span className="text-[10px] uppercase font-bold tracking-wider text-slate-500 dark:text-slate-400 mb-1">Reference</span>
                            <div
                                className="w-16 h-16 md:w-20 md:h-20 rounded-lg shadow-inner bg-cover bg-center"
                                style={{ backgroundImage: `url(${IMAGE_URL})` }}
                            />
                        </div>
                    )}
                </div>

                {isComplete ? (
                    <div className="flex flex-col items-center animate-bounce-short p-4 bg-green-100 dark:bg-green-900/30 rounded-xl w-full max-w-sm">
                        <Trophy className="h-12 w-12 text-green-500 mb-2" />
                        <h4 className="text-xl font-bold text-green-700 dark:text-green-400">
                            {level === MAX_LEVEL ? "You Beat The Game!" : "Level Complete!"}
                        </h4>
                        <p className="text-sm text-green-600 dark:text-green-500 mb-4">Completed in {moves} moves</p>

                        {level < MAX_LEVEL ? (
                            <Button onClick={handleNextLevel} className="bg-green-600 hover:bg-green-700 w-full mb-2">
                                Next Level ({level + 1}) <ArrowRight className="ml-2 h-4 w-4" />
                            </Button>
                        ) : (
                            <Button onClick={handlePlayAgain} className="bg-green-600 hover:bg-green-700 w-full mb-2">
                                Play Again from Level 1
                            </Button>
                        )}
                        <Button onClick={() => initializePuzzle(level)} variant="outline" className="w-full text-green-700 border-green-300 dark:border-green-800">
                            Replay Level {level}
                        </Button>
                    </div>
                ) : (
                    <div
                        className="relative bg-slate-200 dark:bg-slate-800 border-2 border-slate-300 dark:border-slate-700 rounded-lg overflow-hidden shadow-inner"
                        style={{ width: PUZZLE_SIZE, height: PUZZLE_SIZE }}
                    >
                        {tiles.map((tile, index) => {
                            const x = (tile.currentPos % gridSize) * tileSize;
                            const y = Math.floor(tile.currentPos / gridSize) * tileSize;

                            const bgX = (tile.correctPos % gridSize) * tileSize;
                            const bgY = Math.floor(tile.correctPos / gridSize) * tileSize;

                            const isSelected = selectedTile === index;

                            return (
                                <div
                                    key={tile.id}
                                    onClick={() => handleTileClick(index)}
                                    className={`absolute cursor-pointer flex items-start justify-start transition-all duration-300 ease-in-out border hover:opacity-80
                                        ${isSelected ? 'border-4 border-primary z-10 shadow-lg scale-95' : 'border-slate-300 dark:border-slate-700 hover:border-primary/50'}
                                    `}
                                    style={{
                                        width: tileSize,
                                        height: tileSize,
                                        left: x,
                                        top: y,
                                        backgroundImage: `url(${IMAGE_URL})`,
                                        backgroundPosition: `-${bgX}px -${bgY}px`,
                                        backgroundSize: `${PUZZLE_SIZE}px ${PUZZLE_SIZE}px`,
                                    }}
                                >
                                </div>
                            );
                        })}
                    </div>
                )}
            </div>

            {!isComplete && (
                <Button onClick={() => initializePuzzle(level)} variant="outline" className="mt-4 gap-2">
                    <RefreshCw className="h-4 w-4" /> Shuffle Pieces
                </Button>
            )}
        </Card>
    );
}
