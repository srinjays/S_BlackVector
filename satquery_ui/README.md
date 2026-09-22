# Extracted Framer Hero Section & AI Chat Components

This directory contains the full React + TypeScript source code extracted directly from your **SatQuery AI** Framer project.

---

## 📁 Extracted Files

1. **`HeroSection.tsx`**  
   The complete Hero Section component matching your Framer design system:
   - Font loading for `Bricolage Grotesque` & `Inter Display`
   - Background cloud WebP layer overlays
   - Title, rotated satellite icon (-16°), and sub-headline typography
   - Embedded responsive `AiChatPrompt` container

2. **`AiChatPrompt.tsx`** (originally `Codigo.tsx`)  
   The full interactive ChatGPT-style prompt box component:
   - 40.7 KB of TypeScript / Framer Motion code
   - Microphone, attachment, and send button SVG icons
   - Animated typing effects, focus glows, and dark/light theme properties

3. **`index.ts`**  
   Clean TypeScript exports for simple importing.

---

## 🚀 How to Use in Your Local React / Next.js Project

### 1. Import Component
```tsx
import { HeroSection } from "./framer_export"

export default function App() {
  return (
    <main>
      <HeroSection 
        title="SatQuery AI"
        subtitle="Agentic remote sensing assistant that lets users analyze satellite imagery through simple natural language queries."
      />
    </main>
  )
}
```

### 2. Connect to SatQuery AI Backend (`http://localhost:8000`)
```tsx
async function handleQuery(queryText: string, imageId: string, taskType: string) {
  const response = await fetch("http://localhost:8000/v1/ml/process", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      request_id: crypto.randomUUID(),
      task_type: taskType, // "vqa", "caption", "fusion", or "change"
      query: queryText,
      image_ids: [imageId],
      input_scope: "single"
    })
  });
  
  const data = await response.json();
  console.log("Model Response:", data.facts, data.confidence);
}
```
