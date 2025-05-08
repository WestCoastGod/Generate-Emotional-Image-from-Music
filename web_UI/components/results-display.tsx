"use client"

import { useState } from "react"
import Image from "next/image"
import { useToast } from "@/hooks/use-toast"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { AudioWaveformIcon as Waveform, Download, Share2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useAnalysisStore } from "@/lib/store"

export function ResultsDisplay() {
  const { toast } = useToast()
  const { results, isAnalyzing } = useAnalysisStore()
  const [activeTab, setActiveTab] = useState("image")

  const handleDownload = () => {
    if (!results?.imageUrl) return

    // Create a temporary link element
    const link = document.createElement("a")
    link.href = results.imageUrl
    link.download = "music-visualization.png"
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)

    toast({
      title: "Image downloaded",
      description: "Your visualization has been downloaded successfully.",
    })
  }

  const handleShare = () => {
    if (navigator.share && results?.imageUrl) {
      navigator
        .share({
          title: "My Music Visualization",
          text: "Check out this image generated from my music's emotional analysis!",
          url: results.imageUrl,
        })
        .catch((err) => {
          console.error("Error sharing:", err)
        })
    } else {
      toast({
        title: "Sharing not supported",
        description: "Your browser doesn't support the Web Share API.",
        variant: "destructive",
      })
    }
  }

  if (!results && !isAnalyzing) {
    return null
  }

  return (
    <div className="mt-12 w-full">
      <Card>
        <CardHeader>
          <CardTitle>Analysis Results</CardTitle>
          <CardDescription>Visualization based on the emotional valence and arousal of your music</CardDescription>
        </CardHeader>
        <CardContent>
          {isAnalyzing ? (
            <div className="flex h-64 items-center justify-center">
              <div className="flex flex-col items-center space-y-4">
                <Waveform className="h-12 w-12 animate-pulse text-primary" />
                <p className="text-lg font-medium">Analyzing your music...</p>
              </div>
            </div>
          ) : results ? (
            <div className="space-y-6">
              <Tabs defaultValue="image" value={activeTab} onValueChange={setActiveTab}>
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="image">Generated Image</TabsTrigger>
                  <TabsTrigger value="data">Analysis Data</TabsTrigger>
                </TabsList>
                <TabsContent value="image" className="pt-4">
                  <div className="overflow-hidden rounded-lg">
                    {results.imageUrl ? (
                      <Image
                        src={results.imageUrl || "/placeholder.svg"}
                        alt="Generated visualization"
                        width={800}
                        height={500}
                        className="h-auto w-full object-cover"
                      />
                    ) : (
                      <div className="flex h-64 items-center justify-center bg-muted">
                        <p className="text-muted-foreground">No image generated yet</p>
                      </div>
                    )}
                  </div>

                  <div className="mt-4 flex space-x-2">
                    <Button variant="outline" onClick={handleDownload} disabled={!results.imageUrl}>
                      <Download className="mr-2 h-4 w-4" />
                      Download
                    </Button>
                    <Button variant="outline" onClick={handleShare} disabled={!results.imageUrl}>
                      <Share2 className="mr-2 h-4 w-4" />
                      Share
                    </Button>
                  </div>
                </TabsContent>

                <TabsContent value="data" className="pt-4">
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="rounded-lg bg-muted p-4">
                        <h3 className="mb-2 font-medium">Valence</h3>
                        <div className="text-2xl font-bold">{results.valence?.toFixed(2) || "N/A"}</div>
                        <p className="mt-1 text-sm text-muted-foreground">
                          {results.valence && results.valence > 0.6
                            ? "Positive emotional tone"
                            : results.valence && results.valence < 0.4
                              ? "Negative emotional tone"
                              : "Neutral emotional tone"}
                        </p>
                      </div>

                      <div className="rounded-lg bg-muted p-4">
                        <h3 className="mb-2 font-medium">Arousal</h3>
                        <div className="text-2xl font-bold">{results.arousal?.toFixed(2) || "N/A"}</div>
                        <p className="mt-1 text-sm text-muted-foreground">
                          {results.arousal && results.arousal > 0.6
                            ? "High energy/intensity"
                            : results.arousal && results.arousal < 0.4
                              ? "Low energy/intensity"
                              : "Moderate energy/intensity"}
                        </p>
                      </div>
                    </div>

                    {results.additionalData && (
                      <div className="rounded-lg border p-4">
                        <h3 className="mb-2 font-medium">Additional Analysis</h3>
                        <pre className="whitespace-pre-wrap text-sm">
                          {JSON.stringify(results.additionalData, null, 2)}
                        </pre>
                      </div>
                    )}
                  </div>
                </TabsContent>
              </Tabs>
            </div>
          ) : (
            <div className="flex h-64 items-center justify-center">
              <p className="text-muted-foreground">Upload a music file to see results</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
