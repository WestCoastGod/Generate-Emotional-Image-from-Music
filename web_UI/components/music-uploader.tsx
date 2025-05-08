"use client"

import type React from "react"

import { useState } from "react"
import { Upload, Music } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { analyzeMusic } from "@/lib/analyze-music"

export function MusicUploader() {
  const [file, setFile] = useState<File | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      // Check if file is an audio file
      if (!selectedFile.type.startsWith("audio/")) {
        setError("Please upload an audio file")
        setFile(null)
        return
      }

      setFile(selectedFile)
      setError(null)
    }
  }

  const handleUpload = async () => {
    if (!file) return

    setIsUploading(true)
    setUploadProgress(0)

    try {
      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => {
          if (prev >= 95) {
            clearInterval(progressInterval)
            return prev
          }
          return prev + 5
        })
      }, 200)

      // Call your analyze function
      await analyzeMusic(file)

      clearInterval(progressInterval)
      setUploadProgress(100)

      // Reset after successful upload
      setTimeout(() => {
        setFile(null)
        setIsUploading(false)
        setUploadProgress(0)
      }, 1000)
    } catch (err) {
      setError("Failed to analyze music. Please try again.")
      setIsUploading(false)
      setUploadProgress(0)
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-col items-center justify-center rounded-lg border-2 border-dashed border-muted-foreground/25 p-12 text-center">
        <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-muted">
          <Music className="h-6 w-6" />
        </div>
        <div className="mb-2 text-lg font-semibold">{file ? file.name : "Drag and drop your audio file"}</div>
        <p className="mb-4 text-sm text-muted-foreground">
          {file ? `${(file.size / (1024 * 1024)).toFixed(2)} MB` : "MP3, WAV, FLAC up to 10MB"}
        </p>

        <input
          id="music-file"
          type="file"
          accept="audio/*"
          className="hidden"
          onChange={handleFileChange}
          disabled={isUploading}
        />

        <label htmlFor="music-file">
          <Button variant="outline" disabled={isUploading} className="cursor-pointer" type="button">
            Select File
          </Button>
        </label>
      </div>

      {error && <div className="rounded-md bg-destructive/15 p-3 text-center text-sm text-destructive">{error}</div>}

      {file && !isUploading && (
        <Button onClick={handleUpload} className="w-full" disabled={!file}>
          <Upload className="mr-2 h-4 w-4" />
          Analyze Music
        </Button>
      )}

      {isUploading && (
        <div className="space-y-2">
          <Progress value={uploadProgress} className="h-2 w-full" />
          <p className="text-center text-sm text-muted-foreground">
            {uploadProgress < 100 ? "Analyzing..." : "Analysis complete!"}
          </p>
        </div>
      )}
    </div>
  )
}
