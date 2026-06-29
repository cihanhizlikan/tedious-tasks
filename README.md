# Tedious Tasks

A **C#** tool that automatically classifies text and image files, using **ML.NET** and heuristics.

Tedious Tasks takes the manual chore of sorting and categorising files and hands it to a classifier. Text files and images are sorted by content rather than by name or extension — a trained **ML.NET** model handles image classification, while heuristic rules cover the cases where a model is overkill or unreliable.

## Features
- **Classifies both text and image files** by their actual content
- **Image classification via ML.NET** — Microsoft's machine-learning framework for .NET
- **Heuristic rules** complement the model for fast, deterministic cases
- Automates an otherwise tedious, repetitive sorting task

## Tech stack
C# · .NET · ML.NET

## Building and running
```bash
dotnet build
dotnet run
```
<!-- Add usage details: how files/folders are passed in, and where classified results go. -->

## License
<!-- Add your license here, e.g. MIT or Apache-2.0 -->
