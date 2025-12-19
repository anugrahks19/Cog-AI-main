import { useState } from "react";
import { ChevronRight, Brain, AlertCircle, CheckCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const Demo = () => {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [answers, setAnswers] = useState<number[]>([]);
  const [showResults, setShowResults] = useState(false);

  const questions = [
    {
      id: 1,
      question: "Look at this sequence of numbers for 10 seconds, then select the correct order:",
      sequence: "7, 3, 9, 1, 5",
      options: [
        "7, 3, 9, 1, 5",
        "5, 1, 9, 3, 7",
        "3, 7, 1, 9, 5",
        "1, 3, 5, 7, 9"
      ],
      correct: 0,
      type: "memory"
    },
    {
      id: 2,
      question: "Which shape comes next in this pattern?",
      pattern: "Circle → Square → Triangle → Circle → Square → ?",
      options: ["Circle", "Square", "Triangle", "Diamond"],
      correct: 2,
      type: "pattern"
    },
    {
      id: 3,
      question: "Complete this sentence: 'The bird flew _____ the tree.'",
      options: ["over", "under", "through", "around"],
      correct: 0,
      type: "language"
    }
  ];

  const handleAnswer = (answerIndex: number) => {
    const newAnswers = [...answers, answerIndex];
    setAnswers(newAnswers);

    if (currentQuestion < questions.length - 1) {
      setCurrentQuestion(currentQuestion + 1);
    } else {
      setShowResults(true);
    }
  };

  const calculateScore = () => {
    let correct = 0;
    answers.forEach((answer, index) => {
      if (answer === questions[index].correct) {
        correct++;
      }
    });
    return Math.round((correct / questions.length) * 100);
  };

  const getRiskLevel = (score: number) => {
    if (score >= 80) return { level: "Low Risk", color: "text-green-600", bg: "bg-green-50" };
    if (score >= 60) return { level: "Moderate Risk", color: "text-yellow-600", bg: "bg-yellow-50" };
    return { level: "Higher Risk", color: "text-red-600", bg: "bg-red-50" };
  };

  const resetDemo = () => {
    setCurrentQuestion(0);
    setAnswers([]);
    setShowResults(false);
  };

  if (showResults) {
    const score = calculateScore();
    const risk = getRiskLevel(score);

    return (
      <div className="min-h-screen py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-12">
            <h1 className="text-4xl font-bold text-foreground mb-4">Demo Results</h1>
            <p className="text-lg text-muted-foreground">
              Based on your responses, here's what a typical Cog.ai report might look like:
            </p>
          </div>

          <Card className="shadow-card mb-8">
            <CardHeader className="text-center">
              <div className="w-20 h-20 bg-gradient-hero rounded-full flex items-center justify-center mx-auto mb-4">
                <Brain className="h-10 w-10 text-white" />
              </div>
              <CardTitle className="text-2xl">Cognitive Assessment Report</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Score */}
              <div className="text-center">
                <div className="text-4xl font-bold text-primary mb-2">{score}%</div>
                <p className="text-muted-foreground">Overall Performance Score</p>
              </div>

              {/* Risk Level */}
              <div className={`p-4 rounded-lg ${risk.bg} text-center`}>
                <div className={`text-lg font-semibold ${risk.color} mb-2`}>
                  {risk.level}
                </div>
                <p className="text-sm text-muted-foreground">
                  Based on responses to memory, pattern recognition, and language tasks
                </p>
              </div>

              {/* Breakdown */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-foreground">Performance Breakdown:</h3>
                {questions.map((question, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                    <div>
                      <span className="font-medium capitalize">{question.type} Test</span>
                      <p className="text-sm text-muted-foreground">Question {index + 1}</p>
                    </div>
                    <div className="flex items-center space-x-2">
                      {answers[index] === question.correct ? (
                        <CheckCircle className="h-5 w-5 text-green-600" />
                      ) : (
                        <AlertCircle className="h-5 w-5 text-red-600" />
                      )}
                      <span className="text-sm">
                        {answers[index] === question.correct ? "Correct" : "Incorrect"}
                      </span>
                    </div>
                  </div>
                ))}
              </div>

              {/* Recommendations */}
              <div className="bg-card border rounded-lg p-6">
                <h3 className="text-lg font-semibold text-foreground mb-3">Recommendations:</h3>
                <ul className="space-y-2 text-muted-foreground">
                  <li>• Share these results with your healthcare provider</li>
                  <li>• Consider follow-up cognitive screening in 6 months</li>
                  <li>• Maintain active cognitive engagement through reading and puzzles</li>
                  <li>• Regular physical exercise supports cognitive health</li>
                </ul>
              </div>

              {/* Disclaimer */}
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <p className="text-sm text-yellow-800">
                  <strong>Important:</strong> This is a simplified demo. Actual Cog.ai assessments include many more questions and sophisticated analysis. This demo is not a medical diagnosis.
                </p>
              </div>
            </CardContent>
          </Card>

          <div className="text-center space-y-4">
            <Button onClick={resetDemo} variant="outline" size="lg">
              Try Demo Again
            </Button>
            <div>
              <p className="text-muted-foreground mb-4">
                Ready for the full Cog.ai experience?
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <a href="/contact" className="btn-hero">Request Full Access</a>
                <a href="/features" className="btn-secondary">Learn More About Features</a>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen py-20 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-foreground mb-4">Try Cog.ai Demo</h1>
          <p className="text-lg text-muted-foreground mb-6">
            Experience a simplified version of our cognitive assessment. This demo contains 3 sample questions.
          </p>
          
          {/* Progress */}
          <div className="flex justify-center mb-8">
            <div className="flex space-x-2">
              {questions.map((_, index) => (
                <div
                  key={index}
                  className={`w-3 h-3 rounded-full ${
                    index <= currentQuestion ? 'bg-primary' : 'bg-muted'
                  }`}
                />
              ))}
            </div>
          </div>
        </div>

        {/* Question Card */}
        <Card className="shadow-card mb-8">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-xl">
                Question {currentQuestion + 1} of {questions.length}
              </CardTitle>
              <div className="text-sm text-muted-foreground capitalize">
                {questions[currentQuestion].type} Test
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              <div className="text-lg text-foreground">
                {questions[currentQuestion].question}
              </div>

              {questions[currentQuestion].sequence && (
                <div className="bg-gradient-brain p-6 rounded-lg text-center">
                  <div className="text-2xl font-mono font-bold text-primary mb-2">
                    {questions[currentQuestion].sequence}
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Study this sequence for a moment
                  </p>
                </div>
              )}

              {questions[currentQuestion].pattern && (
                <div className="bg-gradient-brain p-6 rounded-lg text-center">
                  <div className="text-lg font-medium text-primary">
                    {questions[currentQuestion].pattern}
                  </div>
                </div>
              )}

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {questions[currentQuestion].options.map((option, index) => (
                  <Button
                    key={index}
                    variant="outline"
                    size="lg"
                    className="p-6 h-auto text-left justify-start hover:bg-primary hover:text-primary-foreground transition-all duration-300"
                    onClick={() => handleAnswer(index)}
                  >
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 rounded-full bg-muted flex items-center justify-center text-sm font-medium">
                        {String.fromCharCode(65 + index)}
                      </div>
                      <span>{option}</span>
                    </div>
                  </Button>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Instructions */}
        <div className="bg-card border rounded-lg p-6 text-center">
          <p className="text-muted-foreground">
            <strong>Instructions:</strong> Take your time to read each question carefully. Select the answer that seems most appropriate to you. There are no trick questions.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Demo;