import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { getEmbeddings } from "./embeddings";
import { Document } from "@langchain/core/documents";
import { VECTOR_DIR } from "./config";
import fs from "fs";
import path from "path";

// Ensure vector directory exists
if (!fs.existsSync(VECTOR_DIR)) {
    fs.mkdirSync(VECTOR_DIR, { recursive: true });
}

// Helper to get file path for a collection
function getVectorFilePath(collectionName: string): string {
    return path.join(VECTOR_DIR, `${collectionName}.json`);
}

export async function getVectorStore(documents?: Document[], collectionName: string = "default") {
    const embeddings = getEmbeddings();
    const filePath = getVectorFilePath(collectionName);

    if (documents) {
        // Create new from documents
        console.log(`Creating persistent vector store for ${collectionName}`);
        const vectorStore = await MemoryVectorStore.fromDocuments(documents, embeddings);

        // Save to disk
        await saveVectorStore(vectorStore, filePath);
        return vectorStore;
    } else {
        // Load existing
        if (!fs.existsSync(filePath)) {
            throw new Error(`Vector store for ${collectionName} not found at ${filePath}`);
        }

        console.log(`Loading persistent vector store for ${collectionName} from disk`);
        const vectorStore = await loadVectorStore(filePath, embeddings);
        return vectorStore;
    }
}

export function vectorStoreExists(collectionName: string): boolean {
    const filePath = getVectorFilePath(collectionName);
    return fs.existsSync(filePath);
}

// ------------------------------------------------------------------
// Internal Helpers for JSON Persistence (MemoryVectorStore hack)
// ------------------------------------------------------------------

interface SavedVector {
    content: string;
    embedding: number[];
    metadata: Record<string, any>;
}

async function saveVectorStore(store: MemoryVectorStore, filePath: string) {
    // Access internal memoryVectors using 'any' cast
    const memoryVectors = (store as any).memoryVectors as SavedVector[];
    const data = JSON.stringify(memoryVectors);
    await fs.promises.writeFile(filePath, data, 'utf-8');
    console.log(`Saved vector store to ${filePath}`);
}

async function loadVectorStore(filePath: string, embeddings: any): Promise<MemoryVectorStore> {
    const data = await fs.promises.readFile(filePath, 'utf-8');
    const memoryVectors = JSON.parse(data) as SavedVector[];

    // Reconstruct vectors and documents
    const vectors = memoryVectors.map(v => v.embedding);
    // Note: We need to reconstruct Document objects correctly
    const documents = memoryVectors.map(v => new Document({
        pageContent: v.content,
        metadata: v.metadata
    }));

    const store = new MemoryVectorStore(embeddings);
    await store.addVectors(vectors, documents);
    return store;
}
