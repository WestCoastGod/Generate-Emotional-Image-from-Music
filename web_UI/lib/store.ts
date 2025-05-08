import { create } from "zustand"

type AnalysisResults = {
  valence?: number
  arousal?: number
  imageUrl?: string
  additionalData?: any
}

type AnalysisStore = {
  isAnalyzing: boolean
  results: AnalysisResults | null
  setIsAnalyzing: (isAnalyzing: boolean) => void
  setResults: (results: AnalysisResults | null) => void
}

export const useAnalysisStore = create<AnalysisStore>((set) => ({
  isAnalyzing: false,
  results: null,
  setIsAnalyzing: (isAnalyzing) => set({ isAnalyzing }),
  setResults: (results) => set({ results }),
}))
