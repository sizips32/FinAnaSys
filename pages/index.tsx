import { useState } from 'react';
import {
  CurrencyDollarIcon,
  CloudIcon,
  CogIcon,
  ChartPieIcon,
  LightBulbIcon,
  ArrowTrendingUpIcon,
} from '@heroicons/react/24/outline';

interface AnalysisModule {
  id: string;
  name: string;
  description: string;
  icon: React.ElementType;
}

const analysisModules: AnalysisModule[] = [
  {
    id: 'dividend',
    name: 'Dividend Analysis',
    description: 'Comprehensive dividend analysis and forecasting',
    icon: CurrencyDollarIcon,
  },
  {
    id: 'machine',
    name: 'Machine Learning',
    description: 'ML-based market prediction and pattern recognition',
    icon: CogIcon,
  },
  {
    id: 'portfolio',
    name: 'Portfolio Analysis',
    description: 'Portfolio optimization and risk assessment',
    icon: ChartPieIcon,
  },
  {
    id: 'quantum',
    name: 'Quantum Analysis',
    description: 'Quantum computing-based financial modeling',
    icon: LightBulbIcon,
  },
  {
    id: 'volatility',
    name: 'Volatility Analysis',
    description: 'Advanced volatility and risk metrics',
    icon: ArrowTrendingUpIcon,
  },
  {
    id: 'wordcloud',
    name: 'Word Cloud Analysis',
    description: 'Market sentiment analysis through word clouds',
    icon: CloudIcon,
  },
];

export default function Home() {
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const startAnalysis = async (moduleId: string) => {
    setLoading(moduleId);
    setError(null);

    try {
      const response = await fetch('/api/run-streamlit', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ moduleId }),
      });

      if (!response.ok) {
        throw new Error('Failed to start analysis');
      }

      const data = await response.json();
      
      if (data.success) {
        // Streamlit 기본 포트로 새 탭 열기
        window.open('http://localhost:8501', '_blank');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(null);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Financial Analysis System
          </h1>
          <p className="text-xl text-gray-600">
            Advanced analytics powered by AI and quantum computing
          </p>
        </div>

        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {analysisModules.map((module) => {
            const Icon = module.icon;
            return (
              <div
                key={module.id}
                className="bg-white overflow-hidden shadow-lg rounded-lg hover:shadow-xl transition-shadow duration-300"
              >
                <div className="p-6">
                  <div className="flex items-center justify-center w-12 h-12 bg-blue-100 rounded-lg mb-4">
                    <Icon className="h-6 w-6 text-blue-600" />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">
                    {module.name}
                  </h3>
                  <p className="text-gray-600 mb-4">{module.description}</p>
                  <button
                    onClick={() => startAnalysis(module.id)}
                    disabled={loading === module.id}
                    className={`w-full px-4 py-2 rounded-md text-sm font-medium transition-colors duration-200 
                      ${
                        loading === module.id
                          ? 'bg-gray-300 cursor-not-allowed'
                          : 'bg-blue-600 hover:bg-blue-700 text-white'
                      }`}
                  >
                    {loading === module.id ? 'Starting...' : 'Start Analysis'}
                  </button>
                </div>
              </div>
            );
          })}
        </div>

        {error && (
          <div className="mt-6 p-4 bg-red-100 border border-red-400 text-red-700 rounded-lg">
            {error}
          </div>
        )}
      </div>
    </div>
  );
} 
