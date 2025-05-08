import { MusicUploader } from "@/components/music-uploader"
import { ResultsDisplay } from "@/components/results-display"

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-6 md:p-24">
      <div className="z-10 w-full max-w-5xl items-center justify-between font-mono text-sm">
        <h1 className="mb-8 text-center text-4xl font-bold tracking-tight">Music Analysis & Visualization</h1>

        <div className="mb-32 grid text-center lg:mb-0 lg:grid-cols-1 lg:text-left">
          <div className="rounded-lg border bg-card p-8 shadow-sm">
            <h2 className="mb-4 text-2xl font-semibold">Upload Your Music</h2>
            <p className="mb-6 text-muted-foreground">
              Upload an audio file to analyze its emotional valence and arousal, and generate a corresponding image.
            </p>
            <MusicUploader />
          </div>

          <ResultsDisplay />
        </div>
      </div>
    </main>
  )
}
